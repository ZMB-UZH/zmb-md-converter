# type: ignore

import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from zmb_md_converter.convert import convert_hcs_ai_to_tiffs, convert_ht_ai_to_tiffs

_DIM_TOOLTIPS = {
    "field": (
        "Split fields into separate files.\n"
        "Deactivate if the output will be processed with ashlar."
    ),
    "time": "Split timepoints into separate files.",
    "channel": "Split channels into separate files.",
    "plane": "Split z-planes into separate files.",
}


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

        # Per-instrument input/output dirs
        self.hcs_input_dir = tk.StringVar()
        self.hcs_output_dir = tk.StringVar()
        self.ht_input_dir = tk.StringVar()
        self.ht_output_dir = tk.StringVar()

        # HT.ai options
        self.ht_only_2D = tk.BooleanVar()
        self.ht_dims = {
            "field": tk.BooleanVar(value=True),
            "time": tk.BooleanVar(),
            "channel": tk.BooleanVar(),
            "plane": tk.BooleanVar(),
        }
        self.ht_fill_mixed_acquisition = tk.BooleanVar(value=True)

        # HCS.ai options
        self.hcs_only_2D = tk.BooleanVar()
        self.hcs_use_montage = tk.BooleanVar(value=True)
        self.hcs_dims = {
            "field": tk.BooleanVar(value=True),
            "time": tk.BooleanVar(),
            "channel": tk.BooleanVar(),
            "plane": tk.BooleanVar(),
        }

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self._create_widgets()

    # ------------------------------------------------------------------
    # Widget helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _browse_row(parent, row, label_text, var, browse_cmd, tooltip):
        """Label + Entry + Browse button row, all with the same tooltip."""
        label = tk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky=tk.W)
        ToolTip(label, tooltip)

        entry = tk.Entry(parent, textvariable=var, width=50)
        entry.grid(row=row, column=1, padx=5, pady=5)
        ToolTip(entry, tooltip)

        btn = tk.Button(parent, text="Browse", command=browse_cmd)
        btn.grid(row=row, column=2, padx=5, pady=5)
        ToolTip(btn, tooltip)

    @staticmethod
    def _checkbutton(parent, row, text, var, tooltip, padx=5):
        """Single Checkbutton with tooltip."""
        cb = tk.Checkbutton(parent, text=text, variable=var)
        cb.grid(row=row, column=0, padx=padx, pady=5, sticky=tk.W)
        ToolTip(cb, tooltip)

    @staticmethod
    def _dimensions_section(parent, start_row, dims_dict):
        """'Dimensions to split' label + one checkbox per dimension.

        Returns the next available row index.
        """
        tk.Label(parent, text="Dimensions to split into different files:").grid(
            row=start_row, column=0, sticky=tk.W, padx=5
        )
        for i, (dim, var) in enumerate(dims_dict.items()):
            cb = tk.Checkbutton(parent, text=dim.capitalize(), variable=var)
            cb.grid(row=start_row + 1 + i, column=0, padx=20, pady=2, sticky=tk.W)
            ToolTip(cb, _DIM_TOOLTIPS[dim])
        return start_row + 1 + len(dims_dict)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _create_widgets(self):
        notebook = ttk.Notebook(self.root)
        notebook.grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W)

        hcs_frame = ttk.Frame(notebook)
        ht_frame = ttk.Frame(notebook)
        notebook.add(hcs_frame, text="HCS.ai")
        notebook.add(ht_frame, text="HT.ai")

        self._create_hcs_ai_widgets(hcs_frame)
        self._create_ht_ai_widgets(ht_frame)

    def _create_ht_ai_widgets(self, parent):
        tk.Label(
            parent,
            text="For converting data from the old HT.ai microscope only.",
            justify=tk.LEFT,
            foreground="gray",
        ).grid(row=0, column=0, columnspan=3, padx=5, pady=(5, 0), sticky=tk.W)

        self._browse_row(
            parent, 1, "Input Directory:", self.ht_input_dir,
            lambda: self.ht_input_dir.set(filedialog.askdirectory()),
            "Top directory of the MD data to be converted.\n"
            "The first subdirectories should be 'TimePoint_n'.\n"
            "If the data was acquired with the Z_over_Time journal, the "
            "Input Directory should contain multiple MD-plate folders.",
        )
        self._browse_row(
            parent, 2, "Output Directory:", self.ht_output_dir,
            lambda: self.ht_output_dir.set(filedialog.askdirectory()),
            "Directory where the converted files will be stored.\n"
            "Needs to be outside of Input Directory.",
        )

        inner = ttk.Notebook(parent)
        inner.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W)

        ome_frame = ttk.Frame(inner)
        imagej_frame = ttk.Frame(inner)
        inner.add(ome_frame, text="Convert to OME-TIFF")
        inner.add(imagej_frame, text="Convert to ImageJ Hyperstack")

        # OME-TIFF tab
        tk.Label(
            ome_frame,
            text="Convert MD-data to OME-TIFF stacks.\n\nOptions:",
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky=tk.W)

        self._checkbutton(
            ome_frame, 1, "Projections only", self.ht_only_2D,
            "If activated, only projections and single planes will be converted.",
        )

        next_row = self._dimensions_section(ome_frame, 2, self.ht_dims)

        self._checkbutton(
            ome_frame, next_row, "Fill Mixed Acquisition",
            self.ht_fill_mixed_acquisition,
            "Handle how mixed acquisitions are treated.\n"
            "Mixed acquisitions are acquisitions where for some channels not all planes"
            " or timepoints were acquired.\n"
            "- Activate to fill missing planes and timepoints with existing data.\n"
            "- Deactivate to leave the missing planes and timepoints blank.\n"
            "(If the data was exported from MetaXpress, this is already done.)",
        )

        tk.Button(
            ome_frame, text="Convert", command=self._convert_ht_ai_ome_tiff
        ).grid(row=next_row + 1, column=0, padx=5, pady=5)

        # ImageJ Hyperstack tab
        tk.Label(
            imagej_frame,
            text=(
                "This is the legacy mode.\n"
                "(Should work similar to the 'MD2Hyperstack' script.)\n"
                "\n"
                "Key differences to original script:\n"
                "- Projections will not be added to 3D stacks.\n"
                "- More accurate z-spacing estimation.\n"
            ),
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky=tk.W)

        tk.Button(
            imagej_frame, text="Convert", command=self._convert_ht_ai_imagej
        ).grid(row=1, column=0, padx=5, pady=5)

    def _create_hcs_ai_widgets(self, parent):
        self._browse_row(
            parent, 0, "Input Directory:", self.hcs_input_dir,
            lambda: self.hcs_input_dir.set(filedialog.askdirectory()),
            "Acquisition folder for the HCS.ai experiment.\n"
            "This is the folder that contains the .mxprotocol file\n"
            "and the experiment/ subfolders.",
        )
        self._browse_row(
            parent, 1, "Output Directory:", self.hcs_output_dir,
            lambda: self.hcs_output_dir.set(filedialog.askdirectory()),
            "Directory where the converted files will be stored.\n"
            "Needs to be outside of Input Directory.",
        )

        tk.Label(
            parent,
            text="Options:",
            justify=tk.LEFT,
        ).grid(row=2, column=0, sticky=tk.W, padx=5, pady=(5, 0))

        self._checkbutton(
            parent, 3, "Projections only", self.hcs_only_2D,
            "If activated, only 2D projections are written.\n"
            "Uses pre-calculated MIPs when available, otherwise projects the z-stack.",
        )
        self._checkbutton(
            parent, 4, "Use montage (stitched) images, if available", self.hcs_use_montage,
            "Use the stitched montage images instead of individual fields, if available.\n"
            "If no montage data is found, falls back to individual fields.",
        )

        next_row = self._dimensions_section(parent, 5, self.hcs_dims)

        tk.Button(
            parent, text="Convert", command=self._convert_hcs_ai
        ).grid(row=next_row, column=0, padx=5, pady=5)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_dirs(input_str, output_str):
        if not input_str:
            messagebox.showerror("Missing Input", "Please select an Input Directory.")
            return None, None
        if not output_str:
            messagebox.showerror("Missing Output", "Please select an Output Directory.")
            return None, None
        if input_str in output_str:
            messagebox.showerror(
                "Invalid Output Directory",
                "Output directory must be outside of the input directory.",
            )
            return None, None
        return Path(input_str), Path(output_str)

    def _convert_ht_ai_ome_tiff(self):
        input_dir, output_dir = self._validate_dirs(
            self.ht_input_dir.get(), self.ht_output_dir.get()
        )
        if input_dir is None:
            return
        try:
            convert_ht_ai_to_tiffs(
                input_dir=input_dir,
                output_dir=output_dir,
                only_2D=self.ht_only_2D.get(),
                dimensions_to_split=[d for d, v in self.ht_dims.items() if v.get()],
                fill_mixed_acquisition=self.ht_fill_mixed_acquisition.get(),
            )
            messagebox.showinfo(
                "Conversion Complete",
                "MD to OME-TIFF conversion completed successfully.",
            )
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Conversion Error", str(e))

    def _convert_ht_ai_imagej(self):
        input_dir, output_dir = self._validate_dirs(
            self.ht_input_dir.get(), self.ht_output_dir.get()
        )
        if input_dir is None:
            return
        try:
            convert_ht_ai_to_tiffs(
                input_dir=input_dir,
                output_dir=output_dir,
                only_2D=False,
                dimensions_to_split=("field",),
                fill_mixed_acquisition=False,
                legacy_mode=True,
            )
            messagebox.showinfo(
                "Conversion Complete",
                "MD to ImageJ-hyperstack conversion completed successfully.",
            )
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Conversion Error", str(e))

    def _convert_hcs_ai(self):
        input_dir, output_dir = self._validate_dirs(
            self.hcs_input_dir.get(), self.hcs_output_dir.get()
        )
        if input_dir is None:
            return
        try:
            convert_hcs_ai_to_tiffs(
                input_dir=input_dir,
                output_dir=output_dir,
                only_2D=self.hcs_only_2D.get(),
                use_montage=self.hcs_use_montage.get(),
                dimensions_to_split=[d for d, v in self.hcs_dims.items() if v.get()],
            )
            messagebox.showinfo(
                "Conversion Complete",
                "HCS.ai to OME-TIFF conversion completed successfully.",
            )
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Conversion Error", str(e))

    def run(self):  # noqa: D102
        self.root.mainloop()

    def on_closing(self):  # noqa: D102
        print("\nexiting...\n")
        self.root.destroy()


if __name__ == "__main__":
    gui = GUI()
    gui.run()


