import os
import sys
import tkinter as tk
import ttkbootstrap as ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2 as cv
import numpy as np

########## Utils
from util.popupmsg import message_box, confirm_exit
from util.run_models import run_model, load_model


class TkInter_App:
    def __init__(self, root):
        self.root = root
        self.root.title("Demo Detection Tester")
        self.root.geometry("1380x1080")

        # Variables
        self.folder_items = []
        self.image_refs = []
        self.selected_folder_path = ""
        self.points = []
        self.box_start = None
        self.box_end = None

        self.model_img_infos = tk.StringVar(value="Image Infos")

        # --- Save options with separate variables
        self.saveImagesCheck = tk.BooleanVar(value=False)
        self.saveVideoCheck = tk.BooleanVar(value=False)

        def get_app_path():
            if getattr(sys, "frozen", False):
                # Running as exe
                return os.path.dirname(sys.executable)
            else:
                # Running as script
                return os.path.dirname(os.path.abspath(__file__))

        app_path = get_app_path()
        self.weights_folder = os.path.join(app_path, "weights")

        # ✅ Create the folder if it doesn't exist
        os.makedirs(self.weights_folder, exist_ok=True)

        # ✅ Load model list (will be empty if no files yet)
        self.models_list = [
            f for f in os.listdir(self.weights_folder) 
            if not f.startswith(".") and not f.endswith(".txt")
        ]
        # print(self.models_list)
        self.selected_model = tk.StringVar()
        if not self.models_list:
            self.selected_model.set("No models found")  # This is correct!
        else:
            self.selected_model.set("Choose Model")

        # Initialize in your __init__ or setup
        self.confVal = tk.DoubleVar(value=0.25)
        self.confLabelStr = tk.StringVar(value=f"{self.confVal.get():.2f}")

        # Layout
        self.create_layout()

    # ----------- Layout Setup -----------
    def create_layout(self):
        self.base = ttk.Frame(self.root)
        self.base.pack(padx=10, pady=10, fill="both", expand=True)

        row_num = 4
        canvas_rowspan = 4

        for row in range(row_num):
            self.base.rowconfigure(row, weight=1)
        self.base.columnconfigure(0, weight=1)
        self.base.columnconfigure(1, weight=3)

        # ----------------- Top Frame (left column)
        top_frame = ttk.Frame(self.base)
        top_frame.grid(row=0, column=0, sticky="nsew", padx=(5, 0), pady=2)

        # Choose File Frame
        choosefile_frame = ttk.Frame(top_frame, borderwidth=1, relief="solid", padding=5)
        choosefile_frame.grid(row=0, column=0, sticky="nsew", pady=2)

        btn_choose = ttk.Button(
            choosefile_frame, text="Upload", command=self.selected_folder
        )
        btn_choose.pack(padx=5, side="left")

        self.info_label = ttk.Label(choosefile_frame, text="No folder selected", font="Calibri 14 bold")
        self.info_label.pack(padx=5, side="left")

        # --- Style to remove gray highlight after selection
        style = ttk.Style()
        style.layout(
            "Custom.TCombobox",
            [
                ("Combobox.downarrow", {"side": "right", "sticky": ""}),
                (
                    "Combobox.padding",
                    {
                        "sticky": "nswe",
                        "children": [("Combobox.textarea", {"sticky": "nswe"})],
                    },
                ),
            ],
        )

        style.configure(
            "Custom.TCombobox",
            borderwidth=1,
            relief="solid",
            background="white",
            foreground="black",
            fieldbackground="white",
            selectbackground="white",
            selectforeground="black",
            padding=5,
        )

        selection_frame = ttk.Frame(top_frame, borderwidth=1, relief="solid", padding=5)
        selection_frame.grid(row=1, column=0, sticky="nsew", pady=2)

        self.model_combobox = ttk.Combobox(
            selection_frame,
            textvariable=self.selected_model,
            values=self.models_list,
            state="readonly",
            foreground="green",
            style="Custom.TCombobox",  # <-- use custom style
            takefocus=False,
            height=5,
        )
        self.model_combobox.pack(fill="both")
        self.model_combobox.bind("<<ComboboxSelected>>", self.on_model_selected)

        # Checkbutton Frame
        checkbutton_frame = ttk.Frame(top_frame)
        checkbutton_frame.grid(row=2, column=0, sticky="ew", pady=8)

        # --- Confidence Value
        conf_frame = ttk.Frame(checkbutton_frame)
        conf_frame.pack(fill="both", pady=5)

        conf_label = ttk.Label(conf_frame, text="Conf", font="Calibri 16")
        conf_label.pack(fill="both", padx=(0, 5), pady=(0, 2), side="left")

        conf_num = ttk.Label(
            conf_frame, textvariable=self.confLabelStr, font="Calibri 16"
        )
        conf_num.pack(fill="both", padx=(0, 5), pady=(0, 2), side="left")

        conf_input = ttk.Scale(
            conf_frame,
            orient="horizontal",
            variable=self.confVal,
            from_=0,
            to=1,
            length=100,
            command=lambda val: self.confLabelStr.set(f"{float(val):.2f}"),
            bootstyle="info",
        )
        conf_input.pack(pady=(0, 5), fill="x")

        save_frame = ttk.Frame(checkbutton_frame)
        save_frame.pack(fill="x")

        save_images_ch = ttk.Checkbutton(
            save_frame, text="Save Images", variable=self.saveImagesCheck
        )
        save_images_ch.pack(side="left", fill="x", expand=True, padx=2)

        # save_video_ch = ttk.Checkbutton(
        #     save_frame, text="Save as Video", variable=self.saveVideoCheck
        # )
        # save_video_ch.pack(side="left", fill="x", expand=True, padx=2)

        # Run Detection
        self.btn_rundetection = ttk.Button(
            top_frame,
            text="Run",
            command=self.run_detection,
            bootstyle="success",
        )
        self.btn_rundetection.state(["disabled"])
        self.btn_rundetection.grid(row=3, column=0, sticky="ew", pady=5)

        # Show / Clear Buttons
        endbtn_frame = ttk.Frame(top_frame)
        endbtn_frame.grid(row=4, column=0, sticky="ew", pady=2)

        self.btn_clear = ttk.Button(
            endbtn_frame, text="Reset All", command=self.clear_all, bootstyle="danger"
        )
        # self.btn_clear.state(["disabled"])
        self.btn_clear.pack(pady=2, fill="both")

        # ----------------- Canvas (right side)
        canvas_frame = ttk.Frame(self.base)
        canvas_frame.grid(
            row=0, column=1, rowspan=canvas_rowspan, sticky="nsew", pady=5, padx=(0, 5)
        )
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)  # Allow vertical stretch

        # Create canvas
        self.canvas = tk.Canvas(  # Note: use tk.Canvas, not ttk.Canvas for scroll support
            canvas_frame,
            width=1000,
            height=1000,
            bg="white",
            borderwidth=1,
            relief="solid",
            yscrollcommand=lambda *args: self.canvas_scroll.set(*args)
        )
        self.canvas.pack(
            side=tk.LEFT,
            expand=True,
            fill=tk.BOTH,
            ipadx=10,
            ipady=10,
            anchor='nw'
        )

        self.canvas_scroll = ttk.Scrollbar(
            canvas_frame,
            command=self.canvas.yview
        )
        self.canvas_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(yscrollcommand=self.canvas_scroll.set)

        # Scrollregion setup (important for scrolling to work)
        self.canvas.bind("<Configure>", self.update_scrollregion)

        # Mouse and drag events
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    ######### # ----------- Logic Methods -----------
    def on_click(self, event):
        self.points.append([event.x, event.y])
        self.box_start = (event.x, event.y)

    def on_drag(self, event):
        self.box_end = (event.x, event.y)

    def on_release(self, event):
        self.box_end = (event.x, event.y)

    def update_scrollregion(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_model_selected(self, event=None):
        selected = self.selected_model.get()
        if selected != "Choose Model":
            self.btn_rundetection.state(["!disabled"])
        else:
            self.btn_rundetection.state(["disabled"])

    # ================== General Methods

    def clear_all(self):
        self.selected_folder_path = ""
        self.folder_items.clear()
        self.image_refs.clear()
        self.info_label.config(text="No folder selected.")
        self.canvas.delete("all")
        self.saveImagesCheck.set(False)
        self.btn_rundetection.state(['disabled'])
        self.confVal.set(0.25)
        self.confLabelStr.set(f"{self.confVal.get():.2f}")
        if not self.models_list:
            self.selected_model.set("No models found")  # This is correct!
        else:
            self.selected_model.set("Choose Model")

    def selected_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            self.info_label.config(text="No folder selected.")
            self.folder_items.clear()
            return

        self.selected_folder_path = folder
        self.folder_items = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.info_label.config(text=f"Selected: {len(self.folder_items)} items")

    def run_detection(self):
        self.canvas.delete("all")
        self.image_refs.clear()

        self.detected_images = []
        self.image_detections = []

        if not self.folder_items:
            message_box(self.root, "No images to show.", type="warning")
            return

        # Load the model once before starting detection
        self.model = load_model(self.weights_folder, self.selected_model)

        # Setup progress bar UI (your existing code)...
        loading_progress = ttk.Frame(self.base)
        loading_progress.grid(row=3, column=0, pady=10, sticky="sw")

        self.loading_label = ttk.Label(
            loading_progress, text="Loading images...", font=("Arial", 14, "bold")
        )
        self.loading_label.pack(pady=2, fill="both")

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            loading_progress,
            length=230,
            variable=self.progress_var,
            maximum=len(self.folder_items),
        )
        self.progress_bar.pack(pady=2, fill="both")

        self.current_index = 0
        self.thumb_w, self.thumb_h = 200, 200
        self.spacing = 210
        self.start_x, self.start_y = 10, 30

        def get_unique_folder(base_folder):
            if not os.path.exists(base_folder):
                return base_folder

            i = 1
            while True:
                new_folder = f"{base_folder} ({i})"
                if not os.path.exists(new_folder):
                    return new_folder
                i += 1

        self.save_images = self.saveImagesCheck.get()
        self.save_folder = None
        if self.save_images:
            self.save_folder = get_unique_folder(
                os.path.join(os.path.expanduser("~/Downloads"), "Demo Saved")
            )
            os.makedirs(self.save_folder)

        self.popup_enabled = False
        if hasattr(self, "popup") and self.popup.winfo_exists():
            self.popup.destroy()

        self.display_next_image()

    def display_next_image(self):
        self.btn_clear.state(['disabled'])
        self.btn_rundetection.state(['disabled'])

        if self.current_index >= len(self.folder_items):
            # Done!
            self.loading_label.destroy()
            self.progress_bar.destroy()
            self.btn_rundetection.state(['!disabled'])
            self.btn_clear.state(['!disabled'])


            # Allow popup again
            self.popup_enabled = True

            if self.save_images and self.save_folder:
                parent = os.path.basename(
                    os.path.dirname(self.save_folder)
                )  # 'Downloads'
                current = os.path.basename(self.save_folder)  # 'Demo Saved (1)'
                saved_dir = f"{parent}/{current}"
                message_box(
                    self.root,
                    f"All full-resolution images saved in '{saved_dir}' folder.",
                    type="info",
                )
            return

        img_path = self.folder_items[self.current_index]
        try:
            # Open original full-res image with PIL
            full_res_img = Image.open(img_path)

            # Convert PIL Image to OpenCV BGR format
            open_cv_image = cv.cvtColor(np.array(full_res_img), cv.COLOR_RGB2BGR)

            # Run detection with YOLO model
            conf = round(self.confVal.get(), 2)
            # print(conf)
            results = run_model(self.model, open_cv_image, myconf=conf)

            # Infos
            info_lines = []
            class_counts = {}

            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    class_name = r.names[class_id]
                    conf_score = box.conf[0].item() * 100

                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    info_lines.append(
                        f"{class_name} {class_counts[class_name]} - {conf_score:.2f}% Conf"
                    )

            info_text = (
                "Detections:\n" + " | ".join(info_lines)
                if info_lines
                else "No detections"
            )
            self.image_detections.append(info_text)

            # plot IMage
            plot_img_cv = results[0].plot()
            plot_img_rgb = cv.cvtColor(plot_img_cv, cv.COLOR_BGR2RGB)
            full_detected_img = Image.fromarray(
                plot_img_rgb
            )  # Keep full size detected image

            thumb_img = full_detected_img.copy()
            thumb_img.thumbnail((self.thumb_w, self.thumb_h))

            photo = ImageTk.PhotoImage(thumb_img)

            # Store references for popup usage:
            self.image_refs.append(photo)  # for thumbnails on canvas
            self.detected_images.append(
                full_detected_img
            )  # full detection image for popup

            if hasattr(self, "popup") and self.popup.winfo_exists():
                self.update_popup_image()

            self.root.update_idletasks()

            canvas_width = self.canvas.winfo_width()
            padding_x = 13
            padding_y = 10
            text_height = 1  # approximate height of label text

            images_per_row = max(1, canvas_width // (self.thumb_w + padding_x))
            row = self.current_index // images_per_row
            col = self.current_index % images_per_row

            x = padding_x + col * (self.thumb_w + padding_x) + self.thumb_w // 2
            y = padding_y + row * (self.thumb_h + text_height - padding_y - 30)

            self.canvas.create_text(
                x,
                y + 10,
                text=f"Image {self.current_index + 1}",
                font=("Arial", 16),
                fill="black",
            )
            img_id = self.canvas.create_image(x, y + 25, anchor="n", image=photo)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            self.canvas.tag_bind(
                img_id,
                "<Button-1>",
                lambda e, i=self.current_index: (
                    self.open_image_popup(i)
                    if getattr(self, "popup_enabled", True)
                    else None
                ),
            )

            if self.save_images:
                save_name = (
                    f"image_{self.current_index + 1}{os.path.splitext(img_path)[1]}"
                )
                save_path = os.path.join(self.save_folder, save_name)
                full_detected_img.save(save_path)

            self.progress_var.set(self.current_index + 1)

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

        self.current_index += 1
        self.root.after(500, self.display_next_image)

    def open_image_popup(self, img_index):
        self.popup_current_index = img_index

        self.popup = tk.Toplevel(self.root)
        self.popup.title(f"Image Viewer ({img_index + 1} / {len(self.folder_items)})")
        self.popup.configure(bg="lightgray")

        popup_base = ttk.Frame(self.popup, padding=10)
        popup_base.pack(fill="both", expand=True)

        nav_container = ttk.Frame(popup_base)
        nav_container.grid(row=0, column=0, sticky="nsew")

        for i in range(3):
            nav_container.columnconfigure(i, weight=1)
        nav_container.rowconfigure(0, weight=2)
        nav_container.rowconfigure(1, weight=1)

        # Use detected_images list for popup
        max_popup_size = (800, 600)
        pil_img = self.detected_images[img_index].copy()
        pil_img.thumbnail(max_popup_size)
        self.popup_img = ImageTk.PhotoImage(pil_img)

        self.prev_btn = ttk.Button(
            nav_container, text="⟨ Previous", command=self.popup_prev_image
        )
        self.prev_btn.grid(row=0, rowspan=2, column=0, padx=10, pady=10, sticky="n")
        self.prev_btn.state(
            ["!disabled"] if self.popup_current_index > 0 else ["disabled"]
        )

        self.img_label = ttk.Label(nav_container, image=self.popup_img)
        self.img_label.grid(row=0, column=1, pady=10, sticky="n")
        self.img_label.image = self.popup_img

        self.model_img_infos = tk.StringVar(value=self.image_detections[img_index])
        self.img_infos = ttk.Label(
            nav_container,
            textvariable=self.model_img_infos,
            font="Calibri 14 bold",
            wraplength=700,
            justify="left",
            foreground="red",
        )
        self.img_infos.grid(row=1, column=1, pady=(0, 5), sticky="nw")

        self.next_btn = ttk.Button(
            nav_container, text="Next ⟩", command=self.popup_next_image
        )
        self.next_btn.grid(row=0, rowspan=2, column=2, padx=10, pady=10, sticky="n")
        self.next_btn.state(
            ["!disabled"]
            if self.popup_current_index < len(self.folder_items) - 1
            else ["disabled"]
        )

    def popup_next_image(self):
        if self.popup_current_index < len(self.folder_items) - 1:
            self.popup_current_index += 1
            self.update_popup_image()

    def popup_prev_image(self):
        if self.popup_current_index > 0:
            self.popup_current_index -= 1
            self.update_popup_image()

    def update_popup_image(self):
        self.popup.title(
            f"Image Viewer ({self.popup_current_index + 1} / {len(self.folder_items)})"
        )

        max_popup_size = (800, 600)
        pil_img = self.detected_images[self.popup_current_index].copy()
        pil_img.thumbnail(max_popup_size)
        self.popup_img = ImageTk.PhotoImage(pil_img)
        self.img_label.config(image=self.popup_img)
        self.img_label.image = self.popup_img

        self.model_img_infos.set(self.image_detections[self.popup_current_index])

        self.prev_btn.state(
            ["!disabled"] if self.popup_current_index > 0 else ["disabled"]
        )
        self.next_btn.state(
            ["!disabled"]
            if self.popup_current_index < len(self.folder_items) - 1
            else ["disabled"]
        )


if __name__ == "__main__":
    root = ttk.Window(themename="cosmo")
    app = TkInter_App(root)
    
    root.protocol("WM_DELETE_WINDOW", lambda: confirm_exit(root))
    root.mainloop()
