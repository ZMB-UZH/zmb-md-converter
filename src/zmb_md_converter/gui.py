# type: ignore

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from zmb_md_converter.convert import (
    convert_md_to_imagej_hyperstacks,
    convert_md_to_ome_tiffs,
)


class ToolTip:  # noqa: D101
    def __init__(self, widget, text):  # noqa: D107
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        widget.bind("<Enter>", self.show_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):  # noqa: D102
        if self.tooltip_window or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("tahoma", "8", "normal"),
        )
        label.pack(ipadx=1)

    def hide_tooltip(self, event):  # noqa: D102
        tw = self.tooltip_window
        self.tooltip_window = None
        if tw:
            tw.destroy()


class GUI:  # noqa: D101
    def __init__(self):  # noqa: D107
        self.root = tk.Tk()
        self.root.title("MD to TIFF-stack Converter")

        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.only_2D = tk.BooleanVar()
        self.dimensions_to_split = {
            "field": tk.BooleanVar(value=True),
            "time": tk.BooleanVar(),
            "channel": tk.BooleanVar(),
            "plane": tk.BooleanVar(),
        }
        self.fill_mixed_acquisition = tk.BooleanVar(value=True)

        # Bind the close event to a custom function
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self._create_widgets()

    def _create_widgets(self):
        # INPUT AND OUTPUT DIRECTORIES
        input_tooltip = (
            "Top directory of the MD data to be converted.\n"
            "The first subdirectories should be 'TimePoint_n'.\n"
            "If the data was acquired with the Z_over_Time journal, the "
            "Input Directory should contain multiple MD-plate folders."
        )
        input_label = tk.Label(self.root, text="Input Directory:")
        input_label.grid(row=0, column=0, sticky=tk.W)
        ToolTip(input_label, input_tooltip)

        input_entry = tk.Entry(self.root, textvariable=self.input_dir, width=50)
        input_entry.grid(row=0, column=1, padx=5, pady=5)
        ToolTip(input_entry, input_tooltip)

        input_button = tk.Button(
            self.root, text="Browse", command=self._browse_input_dir
        )
        input_button.grid(row=0, column=2, padx=5, pady=5)
        ToolTip(input_button, input_tooltip)

        output_tooltip = (
            "Directory where the converted files will be stored.\n"
            "Needs to be outside of Input Directory."
        )
        output_label = tk.Label(self.root, text="Output Directory:")
        output_label.grid(row=1, column=0, sticky=tk.W)
        ToolTip(output_label, output_tooltip)

        output_entry = tk.Entry(self.root, textvariable=self.output_dir, width=50)
        output_entry.grid(row=1, column=1, padx=5, pady=5)
        ToolTip(output_entry, output_tooltip)

        output_button = tk.Button(
            self.root, text="Browse", command=self._browse_output_dir
        )
        output_button.grid(row=1, column=2, padx=5, pady=5)
        ToolTip(output_button, output_tooltip)

        # CREATE A NOTEBOOK FOR TABS
        notebook = ttk.Notebook(self.root)
        notebook.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W)

        # Create frames for each tab
        ome_tiff_frame = ttk.Frame(notebook)
        imagej_frame = ttk.Frame(notebook)

        # Add frames to notebook
        notebook.add(ome_tiff_frame, text="Convert to OME-TIFF")
        notebook.add(imagej_frame, text="Convert to ImageJ Hyperstack")

        # --------------------------------------
        # OME-TIFF FRAME
        # --------------------------------------

        # Text field
        ome_tiff_label = tk.Label(
            ome_tiff_frame,
            text=("Convert MD-data to OME-TIFF stacks.\n\n" "Options:"),
            justify=tk.LEFT,
        )
        ome_tiff_label.grid(row=0, column=0, sticky=tk.W)

        # Only 2D
        only_2D_checkbox = tk.Checkbutton(
            ome_tiff_frame, text="Convert only 2D data", variable=self.only_2D
        )
        only_2D_checkbox.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ToolTip(
            only_2D_checkbox,
            "If activated, only projections and single planes will be converted.",
        )

        # Dimensions to split into different files
        dimensions_label = tk.Label(
            ome_tiff_frame, text="Dimensions to split into different files:"
        )
        dimensions_label.grid(row=2, column=0, sticky=tk.W)
        tooltips = {
            "field": (
                "Split fields into separate files.\n"
                "Deactivate if the output will be processed with ashlar."
            ),
            "time": "Split timepoints into separate files.",
            "channel": "Split channels into separate files.",
            "plane": "Split z-planes into separate files.",
        }
        for i, (dimension, var) in enumerate(self.dimensions_to_split.items()):
            checkbox = tk.Checkbutton(
                ome_tiff_frame, text=dimension.capitalize(), variable=var
            )
            checkbox.grid(row=i + 3, column=0, padx=20, pady=2, sticky=tk.W)
            ToolTip(checkbox, tooltips[dimension])

        # Fill Mixed Acquisition
        fill_mixed_acquisition_checkbox = tk.Checkbutton(
            ome_tiff_frame,
            text="Fill Mixed Acquisition",
            variable=self.fill_mixed_acquisition,
        )
        fill_mixed_acquisition_checkbox.grid(
            row=len(self.dimensions_to_split) + 4, column=0, padx=5, pady=5, sticky=tk.W
        )
        ToolTip(
            fill_mixed_acquisition_checkbox,
            "Handle how mixed acquisitions are treated.\n"
            "Mixed acquisitions are acquisitions where for some channels not all planes"
            " or timepoints were acquired.\n"
            "- Activate to fill missing planes and timepoints with existing data.\n"
            "- Deactivate to leave the missing planes and timepoints blank.\n"
            "(If the data was exported from MetaXpress, this is already done.)",
        )

        # convert button
        convert_button_ome_tiff = tk.Button(
            ome_tiff_frame, text="Convert", command=self._convert_ome_tiff
        )
        convert_button_ome_tiff.grid(
            row=len(self.dimensions_to_split) + 5,
            column=0,
            padx=5,
            pady=5,
        )

        # --------------------------------------
        # IMAGEJ-HYPERSTACK FRAME
        # --------------------------------------

        # Text field
        imagej_label = tk.Label(
            imagej_frame,
            text=(
                "This is the legacy mode.\n"
                "(Should work similar to the 'MD2Hyperstack' script.)\n"
                "\n"
                "Key differences to original script:\n"
                "- Projections will not be added to 3D stacks.\n"
                "- More accurate z-spacing estimation."
                "\n"
            ),
            justify=tk.LEFT,
        )
        imagej_label.grid(row=0, column=0, sticky=tk.W)

        # convert button
        convert_button_imagej = tk.Button(
            imagej_frame,
            text="Convert",
            command=self._convert_imagej,
        )
        convert_button_imagej.grid(row=1, column=0, padx=5, pady=5)

    def _browse_input_dir(self):
        directory = filedialog.askdirectory()
        self.input_dir.set(directory)

    def _browse_output_dir(self):
        directory = filedialog.askdirectory()
        self.output_dir.set(directory)

    def _convert_ome_tiff(self):
        input_dir = self.input_dir.get()
        output_dir = self.output_dir.get()
        if input_dir in output_dir:
            messagebox.showerror(
                "Invalid Output Directory",
                "Output directory must be outside of the input directory.",
            )
            return
        only_2D = self.only_2D.get()
        dimensions_to_split = [
            dimension
            for dimension, var in self.dimensions_to_split.items()
            if var.get()
        ]
        fill_mixed_acquisition = self.fill_mixed_acquisition.get()

        try:
            convert_md_to_ome_tiffs(
                input_dir=Path(input_dir),
                output_dir=Path(output_dir),
                only_2D=only_2D,
                dimensions_to_split=dimensions_to_split,
                fill_mixed_acquisition=fill_mixed_acquisition,
            )
            messagebox.showinfo(
                "Conversion Complete",
                "MD to OME-TIFF conversion completed successfully.",
            )
        except Exception as e:
            messagebox.showerror("Conversion Error", str(e))

    def _convert_imagej(self):
        input_dir = self.input_dir.get()
        output_dir = self.output_dir.get()
        if input_dir in output_dir:
            messagebox.showerror(
                "Invalid Output Directory",
                "Output directory must be outside of the input directory.",
            )
            return

        try:
            convert_md_to_imagej_hyperstacks(
                input_dir=Path(input_dir), output_dir=Path(output_dir)
            )
            messagebox.showinfo(
                "Conversion Complete",
                "MD to ImageJ-hyperstack conversion completed successfully.",
            )
        except Exception as e:
            messagebox.showerror("Conversion Error", str(e))

    def run(self):  # noqa: D102
        self.root.mainloop()

    def on_closing(self):  # noqa: D102
        print("\nexiting...\n")
        self.root.destroy()


if __name__ == "__main__":
    gui = GUI()
    gui.run()
