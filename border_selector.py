import sys
import rawpy
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.path import Path
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
    QComboBox,
    QHBoxLayout,
    QSlider,
    QCheckBox,
)
from PyQt5.QtCore import Qt

PARAMS_CONVERSION = {
    "demosaic_algorithm": rawpy.DemosaicAlgorithm.AHD,
    "half_size": True,
    "use_camera_wb": True,
    "highlight_mode": rawpy.HighlightMode.Clip,
    "output_color": rawpy.ColorSpace.sRGB,
    "dcb_enhance": False,
    "output_bps": 8,
    "gamma": (2.2, 0),
    "exp_shift": 4,
    "no_auto_bright": True,
    "no_auto_scale": True,
    "use_auto_wb": False,
    "user_wb": [1, 1, 1, 1],
}


def debayer_image(image_path):
    with rawpy.imread(image_path) as raw:
        return raw.postprocess(**PARAMS_CONVERSION)


def convert_to_bw(image):
    """Convert an RGB image to black and white without resizing."""
    bw_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    return bw_image


class BorderSelectorApp(QMainWindow):
    def __init__(self, bw_image):
        super().__init__()
        self.bw_image = bw_image
        self.aspect_ratio = 3 / 2
        self.rotation_angle = 0
        self.selected_rotation_angle = None
        self.free_form = False

        # Initialize box parameters
        h, w = self.bw_image.shape
        self.box_center = (w // 2, h // 2)
        self.box_width = w // 3
        self.box_height = int(self.box_width / self.aspect_ratio)

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Border Width Selector")
        self.setGeometry(100, 100, 800, 600)

        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)

        layout = QVBoxLayout(main_widget)

        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.imshow(self.bw_image, cmap="gray")
        self.ax.axis("off")  # Remove axes
        self.canvas = FigureCanvas(self.figure)
        self.canvas.resize_event = self.on_resize
        layout.addWidget(self.canvas)

        control_layout = QVBoxLayout()
        layout.addLayout(control_layout)

        aspect_ratio_layout = QHBoxLayout()
        control_layout.addLayout(aspect_ratio_layout)

        self.aspect_ratio_combo = QComboBox()
        self.aspect_ratio_combo.addItems(["1:1", "3:2", "4:3", "16:9", "16:10"])
        self.aspect_ratio_combo.setCurrentText("3:2")
        self.aspect_ratio_combo.currentTextChanged.connect(self.update_aspect_ratio)
        aspect_ratio_layout.addWidget(QLabel("Aspect Ratio (Width:Height)"))
        aspect_ratio_layout.addWidget(self.aspect_ratio_combo)

        self.free_form_checkbox = QCheckBox("Free Form")
        self.free_form_checkbox.stateChanged.connect(self.toggle_free_form)
        aspect_ratio_layout.addWidget(self.free_form_checkbox)

        rotation_slider_layout = QHBoxLayout()
        control_layout.addLayout(rotation_slider_layout)

        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setRange(-250, 250)  # Allow decimal by scaling
        self.rotation_slider.setValue(self.rotation_angle * 10)
        self.rotation_slider.setSingleStep(1)
        self.rotation_slider.valueChanged.connect(self.update_lines)
        rotation_slider_layout.addWidget(QLabel("Rotation Angle (°)"))
        rotation_slider_layout.addWidget(self.rotation_slider)

        self.rotation_slider_label = QLabel(f"{self.rotation_slider.value() / 10}°")
        rotation_slider_layout.addWidget(self.rotation_slider_label)

        self.set_button = QPushButton("Set Rotation")
        self.set_button.clicked.connect(self.set_parameters)
        control_layout.addWidget(self.set_button)

        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.dragging = False
        self.resizing = False
        self.resize_corner = None
        self.resize_side = None

        self.update_lines()

    def toggle_free_form(self, state):
        self.free_form = state == Qt.Checked
        self.update_aspect_ratio()  # Ensure the aspect ratio is updated correctly
        self.update_lines()

    def update_aspect_ratio(self):
        if not self.free_form:
            aspect_ratio_str = self.aspect_ratio_combo.currentText()
            self.aspect_ratio = {
                "1:1": 1 / 1,
                "3:2": 3 / 2,
                "4:3": 4 / 3,
                "16:9": 16 / 9,
                "16:10": 16 / 10,
            }[aspect_ratio_str]
        self.update_lines()

    def update_lines(self):
        self.rotation_angle = self.rotation_slider.value() / 10
        self.rotation_slider_label.setText(f"{self.rotation_angle}°")
        self.draw_box()

    def draw_box(self):
        h, w = self.bw_image.shape

        cx, cy = self.box_center
        new_w = self.box_width
        new_h = self.box_height if self.free_form else int(new_w / self.aspect_ratio)

        start_x = cx - new_w // 2
        start_y = cy - new_h // 2
        end_x = cx + new_w // 2
        end_y = cy + new_h // 2

        self.ax.clear()
        self.ax.imshow(self.bw_image, cmap="gray")
        self.ax.axis("off")  # Remove axes

        # Calculate the rotated rectangle coordinates
        corners = np.array(
            [[start_x, start_y], [end_x, start_y], [end_x, end_y], [start_x, end_y]]
        )
        angle_rad = np.deg2rad(self.rotation_angle)
        rotation_matrix = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )
        rotated_corners = np.dot(corners - [cx, cy], rotation_matrix) + [cx, cy]

        # Clip coordinates to image boundaries
        rotated_corners[:, 0] = np.clip(rotated_corners[:, 0], 0, w - 1)
        rotated_corners[:, 1] = np.clip(rotated_corners[:, 1], 0, h - 1)

        # Draw the rotated rectangle with transparency
        self.ax.plot(
            [rotated_corners[0, 0], rotated_corners[1, 0]],
            [rotated_corners[0, 1], rotated_corners[1, 1]],
            color="red",
            alpha=0.5,
        )  # Top line
        self.ax.plot(
            [rotated_corners[1, 0], rotated_corners[2, 0]],
            [rotated_corners[1, 1], rotated_corners[2, 1]],
            color="red",
            alpha=0.5,
        )  # Right line
        self.ax.plot(
            [rotated_corners[2, 0], rotated_corners[3, 0]],
            [rotated_corners[2, 1], rotated_corners[3, 1]],
            color="red",
            alpha=0.5,
        )  # Bottom line
        self.ax.plot(
            [rotated_corners[3, 0], rotated_corners[0, 0]],
            [rotated_corners[3, 1], rotated_corners[0, 1]],
            color="red",
            alpha=0.5,
        )  # Left line

        self.canvas.draw()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return

        self.dragging = False
        self.resizing = False
        self.resize_corner = None
        self.resize_side = None

        h, w = self.bw_image.shape
        cx, cy = self.box_center
        new_w = self.box_width
        new_h = self.box_height if self.free_form else int(new_w / self.aspect_ratio)

        start_x = cx - new_w // 2
        start_y = cy - new_h // 2
        end_x = cx + new_w // 2
        end_y = cy + new_h // 2

        corners = [
            (start_x, start_y),
            (end_x, start_y),
            (end_x, end_y),
            (start_x, end_y),
        ]

        sides = [
            ((start_x, start_y), (end_x, start_y)),  # Top side
            ((end_x, start_y), (end_x, end_y)),  # Right side
            ((end_x, end_y), (start_x, end_y)),  # Bottom side
            ((start_x, end_y), (start_x, start_y)),  # Left side
        ]

        # Check if the click is close to any corner for resizing
        for i, (corner_x, corner_y) in enumerate(corners):
            if (
                np.hypot(event.xdata - corner_x, event.ydata - corner_y) < 20
            ):  # Increase tolerance
                self.resizing = True
                self.resize_corner = i
                self.fixed_corner = corners[(i + 2) % 4]
                break

        # Check if the click is close to any side for resizing
        if not self.resizing:
            for i, ((x1, y1), (x2, y2)) in enumerate(sides):
                if (
                    np.hypot(event.xdata - (x1 + x2) / 2, event.ydata - (y1 + y2) / 2)
                    < 20
                ):  # Increase tolerance
                    self.resizing = True
                    self.resize_side = i
                    self.fixed_side = sides[(i + 2) % 4]
                    break

        if not self.resizing:
            self.dragging = True
            self.press_event = event

    def on_motion(self, event):
        if event.inaxes != self.ax:
            return

        if self.dragging:
            dx = event.xdata - self.press_event.xdata
            dy = event.ydata - self.press_event.ydata

            h, w = self.bw_image.shape
            cx = min(
                max(self.box_center[0] + dx, self.box_width // 2),
                w - self.box_width // 2,
            )
            cy = min(
                max(self.box_center[1] + dy, self.box_height // 2),
                h - self.box_height // 2,
            )

            self.box_center = (cx, cy)
            self.press_event = event
            self.draw_box()

        if self.resizing:
            h, w = self.bw_image.shape

            if self.resize_corner is not None:
                fx, fy = self.fixed_corner

                if self.resize_corner == 0 or self.resize_corner == 3:
                    new_w = max(20, abs(fx - event.xdata))
                else:
                    new_w = max(20, abs(event.xdata - fx))

                if not self.free_form:
                    new_h = int(new_w / self.aspect_ratio)
                    new_w = int(new_h * self.aspect_ratio)
                else:
                    if self.resize_corner == 0 or self.resize_corner == 1:
                        new_h = max(20, abs(fy - event.ydata))
                    else:
                        new_h = max(20, abs(event.ydata - fy))

                # Ensure the box stays within image boundaries
                new_w = min(new_w, w)
                new_h = min(new_h, h)

                self.box_width = new_w
                self.box_height = new_h
                self.box_center = ((fx + event.xdata) / 2, (fy + event.ydata) / 2)

            elif self.resize_side is not None:
                ((x1, y1), (x2, y2)) = self.fixed_side

                if self.resize_side == 0:  # Top
                    new_h = max(20, abs(y1 - event.ydata))
                    new_w = int(
                        new_h * self.aspect_ratio
                        if not self.free_form
                        else self.box_width
                    )
                    self.box_center = (self.box_center[0], y2 - new_h / 2)
                elif self.resize_side == 1:  # Right
                    new_w = max(20, abs(x1 - event.xdata))
                    new_h = int(
                        new_w / self.aspect_ratio
                        if not self.free_form
                        else self.box_height
                    )
                    self.box_center = (x2 - new_w / 2, self.box_center[1])
                elif self.resize_side == 2:  # Bottom
                    new_h = max(20, abs(event.ydata - y2))
                    new_w = int(
                        new_h * self.aspect_ratio
                        if not self.free_form
                        else self.box_width
                    )
                    self.box_center = (self.box_center[0], y1 + new_h / 2)
                elif self.resize_side == 3:  # Left
                    new_w = max(20, abs(event.xdata - x2))
                    new_h = int(
                        new_w / self.aspect_ratio
                        if not self.free_form
                        else self.box_height
                    )
                    self.box_center = (x1 + new_w / 2, self.box_center[1])

                # Ensure the box stays within image boundaries
                new_w = min(new_w, w)
                new_h = min(new_h, h)

                self.box_width = new_w
                self.box_height = new_h

            self.draw_box()

    def on_release(self, event):
        self.dragging = False
        self.resizing = False

    def on_resize(self, event):
        self.ax.clear()
        self.ax.imshow(self.bw_image, cmap="gray", aspect="auto")
        self.ax.axis("off")
        self.draw_box()

    def set_parameters(self):
        self.selected_rotation_angle = self.rotation_slider.value() / 10
        self.close()

    def get_delimiter_mask(self):
        h, w = self.bw_image.shape
        cx, cy = self.box_center
        new_w = self.box_width
        new_h = self.box_height if self.free_form else int(new_w / self.aspect_ratio)

        start_x = cx - new_w // 2
        start_y = cy - new_h // 2
        end_x = cx + new_w // 2
        end_y = cy + new_h // 2

        corners = np.array(
            [[start_x, start_y], [end_x, start_y], [end_x, end_y], [start_x, end_y]]
        )
        angle_rad = np.deg2rad(self.rotation_angle)
        rotation_matrix = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )
        rotated_corners = np.dot(corners - [cx, cy], rotation_matrix) + [cx, cy]

        mask = np.zeros((h, w), dtype=bool)
        rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        points = np.vstack((cc.ravel(), rr.ravel())).T

        path = Path(rotated_corners)
        mask = path.contains_points(points).reshape(h, w)

        # Scale up the mask to match the original image size
        mask_upscaled = np.kron(mask, np.ones((2, 2), dtype=bool))

        return mask_upscaled


def select_border_width(image_path):
    image = debayer_image(image_path)
    bw_image = convert_to_bw(image)

    app = QApplication(sys.argv)
    main_window = BorderSelectorApp(bw_image)
    main_window.show()
    app.exec_()

    mask = main_window.get_delimiter_mask()
    return main_window.selected_rotation_angle, mask


if __name__ == "__main__":
    image_path = "DSC03639.ARW"  # Replace with your image path
    selected_rotation_angle, mask = select_border_width(image_path)
    print(f"Selected Rotation Angle: {selected_rotation_angle}°")

    # Print the size of the mask
    print(f"Size of the mask: {mask.shape}")

    # Print the amount of True and False values in the mask
    print(f"Number of True values (inside the box): {np.sum(mask)}")
    print(f"Number of False values (outside the box): {np.size(mask) - np.sum(mask)}")
