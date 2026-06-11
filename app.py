# -*- coding: utf-8 -*-
"""
CT Volume Viewer
----------------
.npy (3次元テンソル) / DICOM フォルダ / NIfTI (.nii, .nii.gz) を読み込み、
GPU ボリュームレンダリング or 等値面でフォトリアルに表示するシンプルなビューワ。
ウインドウ処理 (WL/WW) はスライダーでリアルタイムに反映される。

起動:  python app.py
"""

import sys
import os

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QFileDialog, QGroupBox, QRadioButton,
    QMessageBox, QFrame, QCheckBox,
)


# 表示用の最大解像度 (各軸)。超える場合は自動間引き。画質を上げたければ大きく。
MAX_DIM = 256


# ---------------------------------------------------------------- loaders ---

def load_npy(path):
    """3次元 NumPy 配列を (z, y, x) とみなして読み込む。spacing は等方 1.0。"""
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"3次元配列が必要です (shape={arr.shape})")
    return _normalize_dtype(arr), (1.0, 1.0, 1.0)


def load_nifti(path):
    import nibabel as nib
    img = nib.load(path)
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:  # 時系列なら先頭ボリューム
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"3次元データが必要です (shape={data.shape})")
    zooms = img.header.get_zooms()[:3]  # (sx, sy, sz)
    arr = np.transpose(data, (2, 1, 0))  # (x,y,z) -> (z,y,x)
    return _normalize_dtype(arr), (float(zooms[0]), float(zooms[1]), float(zooms[2]))


def load_dicom_dir(folder):
    import pydicom
    slices = []
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if not os.path.isfile(p):
            continue
        try:
            ds = pydicom.dcmread(p, force=True)
            _ = ds.pixel_array  # 画素データの無いファイルを除外
            slices.append(ds)
        except Exception:
            continue
    if len(slices) < 2:
        raise ValueError("DICOM スライスが 2 枚以上見つかりません")

    def z_pos(ds):
        if hasattr(ds, "ImagePositionPatient"):
            return float(ds.ImagePositionPatient[2])
        return float(getattr(ds, "InstanceNumber", 0))

    slices.sort(key=z_pos)

    vol = np.stack([s.pixel_array for s in slices]).astype(np.float32)
    s0 = slices[0]
    slope = float(getattr(s0, "RescaleSlope", 1.0))
    intercept = float(getattr(s0, "RescaleIntercept", 0.0))
    vol = vol * slope + intercept  # HU 値へ変換

    px = getattr(s0, "PixelSpacing", [1.0, 1.0])
    sx, sy = float(px[1]), float(px[0])
    zs = [z_pos(s) for s in slices]
    dz = np.median(np.diff(zs))
    sz = float(abs(dz)) if abs(dz) > 1e-6 else float(getattr(s0, "SliceThickness", 1.0))
    return vol.astype(np.float32), (sx, sy, sz)


def _normalize_dtype(arr):
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8)
    if arr.dtype in (np.float64, np.float16, np.int64, np.int32):
        return arr.astype(np.float32)
    return arr


def downsample(arr_zyx, spacing, max_dim=MAX_DIM):
    """各軸が max_dim を超える場合はストライド間引きし、spacing を補正する。"""
    nz, ny, nx = arr_zyx.shape
    kz = max(1, int(np.ceil(nz / max_dim)))
    ky = max(1, int(np.ceil(ny / max_dim)))
    kx = max(1, int(np.ceil(nx / max_dim)))
    if (kz, ky, kx) == (1, 1, 1):
        return arr_zyx, spacing, False
    arr2 = np.ascontiguousarray(arr_zyx[::kz, ::ky, ::kx])
    sx, sy, sz = spacing
    return arr2, (sx * kx, sy * ky, sz * kz), True


def make_grid(arr_zyx, spacing):
    """(z,y,x) 配列から pyvista ImageData を作る。"""
    nz, ny, nx = arr_zyx.shape
    grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=spacing)
    # C 順で ravel すると x が最も速く変化する = VTK のポイント順
    grid.point_data["values"] = np.ascontiguousarray(arr_zyx).ravel(order="C")
    return grid


# ------------------------------------------------------------------- GUI ----

PRESETS = [
    ("骨",   300, 1500),
    ("肺",  -600, 1500),
    ("軟部",  40,  400),
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CT Volume Viewer")
        self.resize(1280, 800)

        self.grid = None
        self.grid_lo = None      # しきい値ドラッグ中のプレビュー用 (1/2 解像度)
        self.raw_arr = None      # 読み込んだ元データ (解像度切替用に保持)
        self.raw_spacing = None
        self.volume_actor = None
        self.mesh_actor = None
        self.data_min = 0.0
        self.data_max = 1.0

        # 等値面再計算の間引き用タイマー
        self._iso_timer = QTimer(self)
        self._iso_timer.setSingleShot(True)
        self._iso_timer.setInterval(250)
        self._iso_timer.timeout.connect(self._rebuild_isosurface)

        self._build_ui()

    # ---------------------------------------------------------------- UI ----
    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        # --- 左パネル ---
        panel = QWidget()
        panel.setFixedWidth(280)
        lay = QVBoxLayout(panel)

        btn_npy = QPushButton("NumPy (.npy) を開く")
        btn_dcm = QPushButton("DICOM フォルダを開く")
        btn_nii = QPushButton("NIfTI (.nii/.nii.gz) を開く")
        btn_npy.clicked.connect(self.open_npy)
        btn_dcm.clicked.connect(self.open_dicom)
        btn_nii.clicked.connect(self.open_nifti)
        for b in (btn_npy, btn_dcm, btn_nii):
            lay.addWidget(b)

        self.info_label = QLabel("データ未読み込み")
        self.info_label.setWordWrap(True)
        lay.addWidget(self.info_label)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        lay.addWidget(line)

        # 表示モード
        mode_box = QGroupBox("表示モード")
        mlay = QVBoxLayout(mode_box)
        self.rb_volume = QRadioButton("ボリュームレンダリング")
        self.rb_surface = QRadioButton("等値面 (サーフェス)")
        self.rb_volume.setChecked(True)
        self.rb_volume.toggled.connect(self._on_mode_changed)
        mlay.addWidget(self.rb_volume)
        mlay.addWidget(self.rb_surface)
        lay.addWidget(mode_box)

        # 表示解像度 (各軸の最大ボクセル数)
        res_box = QGroupBox("表示解像度")
        rlay = QHBoxLayout(res_box)
        self.res_buttons = []
        for label, dim in (("低", 128), ("中", 256), ("高", 384)):
            rb = QRadioButton(label)
            rb.setChecked(dim == 256)
            rb._dim = dim
            rb.toggled.connect(self._on_res_changed)
            self.res_buttons.append(rb)
            rlay.addWidget(rb)
        lay.addWidget(res_box)

        # ウインドウ処理
        self.win_box = QGroupBox("ウインドウ処理")
        wlay = QVBoxLayout(self.win_box)

        self.wl_label = QLabel("レベル (WL): 0")
        self.wl_slider = QSlider(Qt.Horizontal)
        self.ww_label = QLabel("幅 (WW): 0")
        self.ww_slider = QSlider(Qt.Horizontal)
        self.op_label = QLabel("不透明度: 25%")
        self.op_slider = QSlider(Qt.Horizontal)
        self.op_slider.setRange(1, 100)
        self.op_slider.setValue(25)

        self.wl_slider.valueChanged.connect(self._on_window_changed)
        self.ww_slider.valueChanged.connect(self._on_window_changed)
        self.op_slider.valueChanged.connect(self._on_window_changed)

        for w in (self.wl_label, self.wl_slider, self.ww_label,
                  self.ww_slider, self.op_label, self.op_slider):
            wlay.addWidget(w)

        # CT プリセット
        wlay.addWidget(QLabel("CT プリセット (HU):"))
        prow = QHBoxLayout()
        for name, wl, ww in PRESETS:
            pb = QPushButton(name)
            pb.clicked.connect(lambda _=False, a=wl, b=ww: self.apply_preset(a, b))
            prow.addWidget(pb)
        wlay.addLayout(prow)
        lay.addWidget(self.win_box)

        # 等値面しきい値
        self.iso_box = QGroupBox("等値面しきい値")
        ilay = QVBoxLayout(self.iso_box)
        self.iso_label = QLabel("しきい値: 0")
        self.iso_slider = QSlider(Qt.Horizontal)
        self.iso_slider.valueChanged.connect(self._on_iso_changed)
        ilay.addWidget(self.iso_label)
        ilay.addWidget(self.iso_slider)
        self.cb_ssao = QCheckBox("SSAO (凹みの陰影・GPU依存)")
        self.cb_ssao.setChecked(False)
        self.cb_ssao.toggled.connect(self._on_ssao_toggled)
        ilay.addWidget(self.cb_ssao)
        self.iso_box.setVisible(False)
        lay.addWidget(self.iso_box)

        btn_reset = QPushButton("カメラをリセット")
        btn_reset.clicked.connect(lambda: self.plotter.reset_camera())
        lay.addWidget(btn_reset)
        lay.addStretch()

        # --- 右: 3D ビュー ---
        self.plotter = QtInteractor(central)
        self.plotter.set_background("#1a1a2e", top="#16213e")
        try:
            self.plotter.enable_lightkit()  # 5灯のライトキットを明示的に有効化
        except Exception:
            pass

        root.addWidget(panel)
        root.addWidget(self.plotter.interactor, stretch=1)
        self.setCentralWidget(central)

    # ------------------------------------------------------------ loading ---
    def open_npy(self):
        path, _ = QFileDialog.getOpenFileName(self, "NumPy ファイルを選択", "", "NumPy (*.npy)")
        if path:
            self._load(load_npy, path)

    def open_nifti(self):
        path, _ = QFileDialog.getOpenFileName(self, "NIfTI ファイルを選択", "", "NIfTI (*.nii *.nii.gz)")
        if path:
            self._load(load_nifti, path)

    def open_dicom(self):
        folder = QFileDialog.getExistingDirectory(self, "DICOM フォルダを選択")
        if folder:
            self._load(load_dicom_dir, folder)

    def _load(self, loader, path):
        try:
            arr, spacing = loader(path)
        except Exception as e:
            QMessageBox.critical(self, "読み込みエラー", str(e))
            return

        # float32 より省メモリ・高速な int16 にできるなら変換 (CT/HU は通常この範囲)
        if (arr.dtype == np.float32 and np.all(np.isfinite(arr))
                and arr.min() >= -32768 and arr.max() <= 32767
                and np.allclose(arr, np.round(arr), atol=0.01)):
            arr = arr.astype(np.int16)
        self.raw_arr = arr
        self.raw_spacing = spacing

        # 外れ値に強いレンジ (0.5/99.5 パーセンタイル)
        self.data_min = float(np.percentile(arr, 0.5))
        self.data_max = float(np.percentile(arr, 99.5))
        if self.data_max <= self.data_min:
            self.data_max = self.data_min + 1.0

        rng = self.data_max - self.data_min
        for s in (self.wl_slider, self.ww_slider, self.iso_slider):
            s.blockSignals(True)
        self.wl_slider.setRange(int(self.data_min), int(self.data_max))
        self.ww_slider.setRange(1, max(2, int(rng)))
        self.iso_slider.setRange(int(self.data_min), int(self.data_max))
        self.wl_slider.setValue(int(self.data_min + rng * 0.5))
        self.ww_slider.setValue(int(rng * 0.5))
        self.iso_slider.setValue(int(self.data_min + rng * 0.5))
        for s in (self.wl_slider, self.ww_slider, self.iso_slider):
            s.blockSignals(False)
        self._on_window_changed()  # ラベル更新
        self._apply_resolution(reset_camera=True)

    def _max_dim(self):
        for rb in self.res_buttons:
            if rb.isChecked():
                return rb._dim
        return MAX_DIM

    def _apply_resolution(self, reset_camera=False):
        """保持している元データから表示解像度に応じたグリッドを再構築する。"""
        if self.raw_arr is None:
            return
        arr, spacing, downsampled = downsample(self.raw_arr, self.raw_spacing,
                                               self._max_dim())
        self.grid = make_grid(arr, spacing)
        # しきい値プレビュー用の 1/2 解像度グリッド
        arr_lo = arr[::2, ::2, ::2]
        sp_lo = (spacing[0] * 2, spacing[1] * 2, spacing[2] * 2)
        self.grid_lo = make_grid(np.ascontiguousarray(arr_lo), sp_lo)

        ds_note = (f"\n表示用に間引き: {self.raw_arr.shape} → {arr.shape}"
                   if downsampled else "")
        self.info_label.setText(
            f"形状: {arr.shape}\n値域: {arr.min():.0f} 〜 {arr.max():.0f}\n"
            f"spacing: ({spacing[0]:.2f}, {spacing[1]:.2f}, {spacing[2]:.2f})" + ds_note
        )
        self._show_current_mode(reset_camera=reset_camera)

    # ---------------------------------------------------------- rendering ---
    def _clear_actors(self):
        # plotter.clear() はライトまで消すことがあるためアクターのみ個別に削除
        if self.volume_actor is not None:
            self.plotter.remove_actor(self.volume_actor, render=False)
        if self.mesh_actor is not None:
            self.plotter.remove_actor(self.mesh_actor, render=False)
        self.volume_actor = None
        self.mesh_actor = None

    def _show_current_mode(self, reset_camera=False):
        if self.grid is None:
            return
        self._clear_actors()
        if self.rb_volume.isChecked():
            self._set_ssao(False)
            self._build_volume()
        else:
            self._set_ssao(self.cb_ssao.isChecked())
            self._rebuild_isosurface()
        if reset_camera:
            self.plotter.reset_camera()
        self.plotter.render()

    def _build_volume(self):
        self.volume_actor = self.plotter.add_volume(
            self.grid, scalars="values", shade=True,
            clim=[self.data_min, self.data_max], show_scalar_bar=False,
        )
        prop = self.volume_actor.prop
        prop.SetAmbient(0.25)
        prop.SetDiffuse(0.9)
        prop.SetSpecular(0.4)
        prop.SetSpecularPower(20)
        prop.SetInterpolationTypeToLinear()

        # 操作中は粗く・静止時は高品質にサンプリングを自動調整
        mapper = self.volume_actor.mapper
        try:
            mapper.SetAutoAdjustSampleDistances(True)
        except AttributeError:
            pass
        # ドラッグ中の目標フレームレート (高いほど操作が軽い)
        try:
            self.plotter.iren.interactor.SetDesiredUpdateRate(60.0)
        except AttributeError:
            pass

        self._update_transfer_functions()

    def _update_transfer_functions(self):
        """WL/WW から色・不透明度の伝達関数を作り直す (GPU 再アップロード無しで高速)。"""
        if self.volume_actor is None:
            return
        wl = float(self.wl_slider.value())
        ww = max(1.0, float(self.ww_slider.value()))
        max_op = self.op_slider.value() / 100.0
        lo, hi = wl - ww / 2.0, wl + ww / 2.0

        prop = self.volume_actor.prop

        ctf = prop.GetRGBTransferFunction()
        ctf.RemoveAllPoints()
        # 暖色寄りのグレースケール (骨・組織が自然に見える)
        ctf.AddRGBPoint(lo, 0.0, 0.0, 0.0)
        ctf.AddRGBPoint(lo + (hi - lo) * 0.5, 0.55, 0.45, 0.38)
        ctf.AddRGBPoint(hi, 1.0, 0.97, 0.92)

        otf = prop.GetScalarOpacity()
        otf.RemoveAllPoints()
        otf.AddPoint(lo, 0.0)
        otf.AddPoint(lo + (hi - lo) * 0.35, max_op * 0.15)
        otf.AddPoint(hi, max_op)
        otf.AddPoint(max(hi, self.data_max), max_op)

        self.plotter.render()

    def _set_ssao(self, on):
        """SSAO (環境光遮蔽)。凹部に陰影が付き等値面の立体感が増す。"""
        try:
            if on and self.grid is not None:
                self.plotter.enable_ssao(radius=self.grid.length * 0.02, blur=True)
            else:
                self.plotter.disable_ssao()
        except Exception:
            pass  # 古いGPU/ドライバで未対応でも動作は継続

    def _rebuild_isosurface(self, preview=False):
        """等値面を再構築。preview=True ならドラッグ中用に粗く高速に。"""
        if self.grid is None or not self.rb_surface.isChecked():
            return
        grid = self.grid_lo if (preview and self.grid_lo is not None) else self.grid
        thr = float(self.iso_slider.value())
        if self.mesh_actor is not None:
            self.plotter.remove_actor(self.mesh_actor, render=False)
            self.mesh_actor = None
        try:
            mesh = grid.contour([thr], scalars="values", method="flying_edges")
        except Exception:
            mesh = grid.contour([thr], scalars="values")
        if mesh.n_points > 0:
            if not preview:
                # 巨大メッシュは間引いて軽量化 (見た目はほぼ変わらない)
                if mesh.n_points > 500_000:
                    try:
                        mesh = mesh.decimate_pro(1.0 - 500_000 / mesh.n_points)
                    except Exception:
                        pass
                # ボクセル段差を軽く平滑化 → 法線が滑らかになり陰影が乗る
                try:
                    mesh = mesh.smooth_taubin(n_iter=30, pass_band=0.05)
                except Exception:
                    pass
            self.mesh_actor = self.plotter.add_mesh(
                mesh, color="#e8dcc8", smooth_shading=True,
                ambient=0.15, diffuse=0.85, specular=0.35, specular_power=15,
                show_scalar_bar=False,
            )
        self.plotter.render()

    # ------------------------------------------------------------- events ---
    def _on_mode_changed(self):
        is_volume = self.rb_volume.isChecked()
        self.win_box.setVisible(is_volume)
        self.iso_box.setVisible(not is_volume)
        self._show_current_mode()

    def _on_window_changed(self):
        self.wl_label.setText(f"レベル (WL): {self.wl_slider.value()}")
        self.ww_label.setText(f"幅 (WW): {self.ww_slider.value()}")
        self.op_label.setText(f"不透明度: {self.op_slider.value()}%")
        self._update_transfer_functions()

    def _on_ssao_toggled(self, on):
        self._set_ssao(on and self.rb_surface.isChecked())
        self.plotter.render()

    def _on_iso_changed(self):
        self.iso_label.setText(f"しきい値: {self.iso_slider.value()}")
        self._rebuild_isosurface(preview=True)  # 粗い即時プレビュー
        self._iso_timer.start()                 # 手を止めたら高解像度で確定

    def _on_res_changed(self, checked):
        if checked:
            self._apply_resolution()

    def apply_preset(self, wl, ww):
        self.wl_slider.setValue(int(np.clip(wl, self.wl_slider.minimum(),
                                            self.wl_slider.maximum())))
        self.ww_slider.setValue(int(np.clip(ww, self.ww_slider.minimum(),
                                            self.ww_slider.maximum())))
        self.iso_slider.setValue(int(np.clip(wl, self.iso_slider.minimum(),
                                             self.iso_slider.maximum())))

    def closeEvent(self, event):
        self.plotter.close()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

