import ttkbootstrap as ttk
# from ttkbootstrap.dialogs import Messagebox


########## Alert
# def message_box(message):
#     Messagebox.ok("No images to show.", title="Message", alert=True)


def message_box(root, message, type):
    if type == "warning":
        bg, fg = "#FFF3CD", "#856404"
    elif type == "error":
        bg, fg = "#F8D7DA", "#721C24"
    elif type == "info":
        bg, fg = "#D1ECF1", "#0C5460"
    else:
        bg, fg = "#FFFFFF", "black"

    custom_message_box(root, type.title(), message, bg_color=bg, fg_color=fg)


def custom_message_box(root, title, message, bg_color=None, fg_color=None):
    popup = ttk.Toplevel(root)
    popup.title(title)
    popup.resizable(False, False)

    # Set size
    width, height = 350, 150

    # Get screen width and height
    screen_width = popup.winfo_screenwidth()
    screen_height = popup.winfo_screenheight()

    # Calculate x and y coordinates for the popup to be centered
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2

    popup.geometry(f"{width}x{height}+{x}+{y}")

    popup.configure(bg=bg_color)

    label = ttk.Label(
        popup,
        text=message,
        background=bg_color,
        foreground=fg_color,
        font=("Segoe UI", 14),
        anchor="center",  # Centers vertically and horizontally in the cell
        justify="center",  # Centers multi-line text horizontally
        wraplength=240,
    )
    label.pack(expand=True, fill="both", padx=20, pady=20)

    def close_popup():
        popup.destroy()

    btn_ok = ttk.Button(popup, text="OK", command=close_popup, bootstyle="primary")
    btn_ok.pack(pady=(0, 15))

    popup.grab_set()
    root.wait_window(popup)


def confirm_exit(root, title="Exit", message="Are you sure you want to exit?"):
    popup = ttk.Toplevel(root)
    popup.title(title)
    popup.resizable(False, False)
    popup.geometry("350x180")
    popup.grab_set()

    screen_width = popup.winfo_screenwidth()
    screen_height = popup.winfo_screenheight()
    x = (screen_width - 350) // 2
    y = (screen_height - 180) // 2
    popup.geometry(f"+{x}+{y}")

    label = ttk.Label(
        popup, 
        text=message, 
        font=("Segoe UI", 14, "bold"),
        anchor="center",     # Centers vertically and horizontally in the cell
        justify="center",    # Centers multi-line text horizontally
    )
    label.pack(expand=True, fill="both", padx=20, pady=(20, 10))

    button_frame = ttk.Frame(popup)
    button_frame.pack(pady=10)

    def exit_app():
        popup.destroy()
        root.destroy()

    def cancel():
        popup.destroy()

    yes_btn = ttk.Button(button_frame, text="Yes", command=exit_app, bootstyle="danger")
    yes_btn.pack(side="left", padx=10)

    no_btn = ttk.Button(button_frame, text="No", command=cancel, bootstyle="secondary")
    no_btn.pack(side="left", padx=10)

    popup.bind("<Escape>", lambda e: cancel())
