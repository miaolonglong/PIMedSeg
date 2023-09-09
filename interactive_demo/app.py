import imp
from operator import mod
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from turtle import Turtle
from typing import Text
from unicodedata import category
import yaml
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from isegm.inference import utils
import torch

import cv2
import numpy as np
import time, math
from PIL import Image

from interactive_demo.canvas import CanvasImage
from interactive_demo.controller import InteractiveController
from interactive_demo.wrappers import BoundedNumericalEntry, FocusHorizontalScale, FocusCheckButton, \
    FocusButton, FocusLabelFrame
import os


class InteractiveDemoApp(ttk.Frame):
    def __init__(self, master, args, model):
        super().__init__(master)
        self.checkpoint_name = 'xxx'
        self.master = master
        self.tdir = ''
        master.title("Interactive Medical Images Segmentation")

        master.withdraw()
        master.update_idletasks()
        x = (master.winfo_screenwidth() - master.winfo_reqwidth()) / 2
        y = (master.winfo_screenheight() - master.winfo_reqheight()) / 2
        master.geometry("+%d+%d" % (x, y))
        self.pack(fill="both", expand=True)

        self.limit_longest_size = args.limit_longest_size

        self.controller = InteractiveController(model, args.device,
                                                predictor_params={'brs_mode': 'NoBRS'},
                                                update_image_callback=self._update_image)
        self.cnt = 0
        self.last_points = None

        self._init_state()
        self._add_menu()
        self._add_canvas()
        self._add_buttons()

        master.bind('<space>', lambda event: self.controller.finish_object())
        master.bind('a', lambda event: self.controller.partially_finish_object())

        self.state['zoomin_params']['skip_clicks'].trace(mode='w', callback=self._reset_predictor)
        self.state['zoomin_params']['target_size'].trace(mode='w', callback=self._reset_predictor)
        self.state['zoomin_params']['expansion_ratio'].trace(mode='w', callback=self._reset_predictor)

    def _init_state(self):
        self.state = {
            'zoomin_params': {
                'use_zoom_in': tk.BooleanVar(value=True),
                'fixed_crop': tk.BooleanVar(value=True),
                'skip_clicks': tk.IntVar(value=-1),
                'target_size': tk.IntVar(value=min(400, self.limit_longest_size)),
                'expansion_ratio': tk.DoubleVar(value=1.4)
            },

            'predictor_params': {
                'net_clicks_limit': tk.IntVar(value=8)
            },
            'brs_mode': tk.StringVar(value='NoBRS'),
            'prob_thresh': tk.DoubleVar(value=0.5),
            'lbfgs_max_iters': tk.IntVar(value=20),

            'alpha_blend': tk.DoubleVar(value=0.5),
            'click_radius': tk.IntVar(value=3),
            'scribble_width': tk.IntVar(value=2),
        }

    def _add_menu(self):
        self.menubar = FocusLabelFrame(self, bd=1)
        self.menubar.pack(side=tk.TOP, fill='x')
        self.menu = tk.Menu(self.menubar)
        self.fmenu = tk.Menu(self.menu)
        self.vmenu = tk.Menu(self.menu)
        self.amenu = tk.Menu(self.menu)
        self.hmenu = tk.Menu(self.menu)
        self.fmenu.add_command(label='Open File...', command=self._load_image_callback)
        self.fmenu.add_command(label='Open Folder...', command=self._load_directory_callback)
        self.fmenu.add_command(label='Close Folder', command=self._close_directory_callback)
        self.fmenu.add_command(label='Exit', command=self.master.quit)
        self.vmenu.add_command(label='Clear Window', command=self.clear_interactive_window)
        self.hmenu.add_command(label='About', command=self._about_callback)
        self.menu.add_cascade(label='File',menu=self.fmenu)
        self.menu.add_cascade(label='View',menu=self.vmenu)
        self.menu.add_cascade(label='Help',menu=self.hmenu)
        self.master['menu'] = self.menu

    def _add_canvas(self):
        
        self.canvas_data_frame = FocusLabelFrame(self, text="Data") 
        self.canvas_data_frame.rowconfigure(0, weight=1)
        self.canvas_data_frame.columnconfigure(0, weight=1)
        self.canvas_data_frame.pack(side=tk.LEFT, fill="both", padx=5, pady=5)
        self.dirsb_axis_y = tk.Scrollbar(self.canvas_data_frame)
        self.dirsb_axis_y.pack(side=tk.RIGHT,fill=tk.Y)
        self.dirsb_axis_x = tk.Scrollbar(self.canvas_data_frame, orient=tk.HORIZONTAL)
        self.dirsb_axis_x.pack(side=tk.BOTTOM,fill=tk.X)
        self.dirs = tk.Listbox(self.canvas_data_frame,height=15,width=20,yscrollcommand=self.dirsb_axis_y.set, xscrollcommand=self.dirsb_axis_x.set, exportselection=False)
        self.dirs.pack(side=tk.LEFT,fill=tk.BOTH, expand=True)
        self.dirs.bind('<<ListboxSelect>>', self.select_image_from_dir_callback)
        self.dirsb_axis_y.config(command=self.dirs.yview)
        self.dirsb_axis_x.config(command=self.dirs.xview)

        self.canvas_frame = FocusLabelFrame(self, text="Interaction Window")
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(1, weight=1)

        self.canvas = tk.Canvas(self.canvas_frame, highlightthickness=0, cursor="hand1", width=500, height=400)
        self.canvas.grid(row=0, column=1, sticky='nswe', padx=5, pady=5)

        self.canvas.bind("<B1-Motion>", self.draw_scribble)

        self.image_on_canvas = None
        self.canvas_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=5, pady=5)

    def _add_buttons(self):
        self.control_frame = FocusLabelFrame(self, text="Management Panel")
        self.control_frame.pack(side=tk.TOP, fill='x', padx=5, pady=5)
        master = self.control_frame

        self.models_options_frame = FocusLabelFrame(master, text='Model')
        self.models_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        self.model_architecture_combobox = ttk.Combobox(self.models_options_frame, textvariable=tk.StringVar(), value=('HRNet-TFineNet','HRNet-OCR','ResNet-50'), exportselection=False)

        self.model_architecture_combobox.current(0)
        self.model_architecture_combobox.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        self.load_model_button = \
            FocusButton(self.models_options_frame, text='Load model parameters...', bg='#b6d7a8', fg='black', width=10, height=2,
                        command=self.load_es_model)
        self.load_model_button.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        self.model_pre_name_label = tk.Label(self.models_options_frame, text='Model name: ')
        self.model_pre_name_label.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        self.model_name_label = tk.Label(self.models_options_frame, text='default')
        self.model_name_label.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)


        self.tags_options_frame = FocusLabelFrame(master, text="Tag")
        self.tags_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        self.tags_options_child_frame = FocusLabelFrame(self.tags_options_frame, height=20)
        self.tags_options_child_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)

        self.tagsb = tk.Scrollbar(self.tags_options_child_frame)
        self.tagsb.pack(side=tk.RIGHT,fill=tk.Y)
        self.tags_options_listboxs = tk.Listbox(self.tags_options_child_frame, height=5, width=25,yscrollcommand=self.tagsb.set, exportselection=False)
        
        self.tags_options_listboxs.pack(side=tk.TOP, fill=tk.X)
        self.tagsb.config(command=self.tags_options_listboxs.yview)
        # to do
        # categories_stream = open('interactive_demo/categories.yaml', encoding='utf-8', mode='r')
        # categories_stream.close()

        self.tag_list = ['Tumor', 'Liver', 'Spleen']
        for tag in self.tag_list:
            self.tags_options_listboxs.insert(tk.END, tag)
        self.tags_options_listboxs.select_set(0)
        
        self.add_tag_button = \
            FocusButton(self.tags_options_frame, text='Add tag', bg='#b6d7a8', fg='black', width=10, height=2,
                        command=self.add_tag)
        self.add_tag_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        self.delete_tag_button = \
            FocusButton(self.tags_options_frame, text='Delete tag', bg='#ea9999', fg='black', width=10, height=2, #A52A2A
                        command=self.delete_tag)
        self.delete_tag_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)

        self.controls_options_frame = FocusLabelFrame(master, text="Controls")
        self.controls_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        self.prediction_threshold_frame = FocusLabelFrame(self.controls_options_frame)
        self.prediction_threshold_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        self.prediction_threshold_label = tk.Label(self.prediction_threshold_frame, text='Prediction threshold', anchor='w')
        self.prediction_threshold_label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.prediction_threshold_frame, from_=0.0, to=1.0, command=self._update_prob_thresh,
                             variable=self.state['prob_thresh']).pack(side=tk.TOP, padx=10, anchor='w')
        self.overlap_diaphaneity_frame = FocusLabelFrame(self.controls_options_frame)
        self.overlap_diaphaneity_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        self.overlap_diaphaneity_label = tk.Label(self.overlap_diaphaneity_frame, text='Mask overlap diaphaneity', anchor='w')
        self.overlap_diaphaneity_label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.overlap_diaphaneity_frame, from_=0.0, to=1.0, command=self._update_blend_alpha,
                             variable=self.state['alpha_blend']).pack(side=tk.TOP, padx=10, anchor='w')
        self.visualisation_control_frame = FocusLabelFrame(self.controls_options_frame)
        self.visualisation_control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        self.click_radius_label = tk.Label(self.visualisation_control_frame, text="Visualisation click radius", anchor='w')
        self.click_radius_label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.visualisation_control_frame, from_=0, to=7, resolution=1, command=self._update_click_radius,
                             variable=self.state['click_radius']).pack(padx=10, anchor='w')
        self.scribble_linewidth_label = tk.Label(self.visualisation_control_frame, text="Visualisation scribble line-width", anchor='w')
        self.scribble_linewidth_label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.visualisation_control_frame, from_=0, to=4, resolution=1, command=self._update_scribble_linewidth,
                             variable=self.state['scribble_width']).pack(padx=10, anchor='w')
        self.annotation_panel_frame = FocusLabelFrame(self.controls_options_frame, text='Annotation')
        self.annotation_panel_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        self.reset_annotation_button = \
            FocusButton(self.annotation_panel_frame, text='Reset', bg='#ea9999', fg='black', width=10, height=2,
                        command=self._reset_last_object)
        
        self.reset_annotation_button.grid(row=0, column=0, padx=10, pady=3)
        self.undo_annotation_button = \
            FocusButton(self.annotation_panel_frame, text='Undo', bg='#ffe599', fg='black', width=10, height=2,
                        command=self.app_undo_interaction)
        
        self.undo_annotation_button.grid(row=0, column=1, padx=10, pady=3)
        self.finish_object_button = \
            FocusButton(self.annotation_panel_frame, text='Finish\nobject', bg='#7FFFD4', fg='black', width=10, height=2,
                        command=self.app_finish_object)
        
        self.finish_object_button.grid(row=1, column=0, padx=10, pady=3)
        self.save_mask_button = \
            FocusButton(self.annotation_panel_frame, text='Save\nmask', bg='#b6d7a8', fg='black', width=10, height=2,
                        command=self._save_mask_callback)
        
        self.save_mask_button.grid(row=1, column=1, padx=10, pady=3)

        self.clicks_options_frame = FocusLabelFrame(master, text="Clicks management")
        
        self.finish_object_button = \
            FocusButton(self.clicks_options_frame, text='Finish\nobject', bg='#b6d7a8', fg='black', width=10, height=2,
                        state=tk.DISABLED, command=self.controller.finish_object)
        self.finish_object_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        self.undo_click_button = \
            FocusButton(self.clicks_options_frame, text='Undo click', bg='#ffe599', fg='black', width=10, height=2,
                        state=tk.DISABLED, command=self.controller.undo_click)
        self.undo_click_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        self.reset_clicks_button = \
            FocusButton(self.clicks_options_frame, text='Reset clicks', bg='#ea9999', fg='black', width=10, height=2,
                        state=tk.DISABLED, command=self._reset_last_object)
        self.reset_clicks_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)

        self.zoomin_options_frame = FocusLabelFrame(master, text="ZoomIn options")
        
        FocusCheckButton(self.zoomin_options_frame, text='Use ZoomIn', command=self._reset_predictor,
                         variable=self.state['zoomin_params']['use_zoom_in']).grid(row=0, column=0, padx=10)
        FocusCheckButton(self.zoomin_options_frame, text='Fixed crop', command=self._reset_predictor,
                         variable=self.state['zoomin_params']['fixed_crop']).grid(row=1, column=0, padx=10)
        tk.Label(self.zoomin_options_frame, text="Skip clicks").grid(row=0, column=1, pady=1, sticky='e')
        tk.Label(self.zoomin_options_frame, text="Target size").grid(row=1, column=1, pady=1, sticky='e')
        tk.Label(self.zoomin_options_frame, text="Expand ratio").grid(row=2, column=1, pady=1, sticky='e')
        BoundedNumericalEntry(self.zoomin_options_frame, variable=self.state['zoomin_params']['skip_clicks'],
                              min_value=-1, max_value=None, vartype=int,
                              name='zoom_in_skip_clicks').grid(row=0, column=2, padx=10, pady=1, sticky='w')
        BoundedNumericalEntry(self.zoomin_options_frame, variable=self.state['zoomin_params']['target_size'],
                              min_value=100, max_value=self.limit_longest_size, vartype=int,
                              name='zoom_in_target_size').grid(row=1, column=2, padx=10, pady=1, sticky='w')
        BoundedNumericalEntry(self.zoomin_options_frame, variable=self.state['zoomin_params']['expansion_ratio'],
                              min_value=1.0, max_value=2.0, vartype=float,
                              name='zoom_in_expansion_ratio').grid(row=2, column=2, padx=10, pady=1, sticky='w')
        self.zoomin_options_frame.columnconfigure((0, 1, 2), weight=1)


    def load_es_model(self):
        self.menubar.focus_set()
        if self._check_entry(self):
            filename = filedialog.askopenfilename(parent=self.master, filetypes=[
                ("Models", "*.pth *.pt"),
                ("All files", "*.*"),
            ], title="Choose a model")

            if len(filename) > 0:
                model = utils.load_is_model(filename, torch.device('cuda:0'), cpu_dist_maps=True)
                self.model_name_label['text'] = os.path.basename(filename)
                self.controller.net = model

    def add_tag(self):
        self.add_tag_button.focus_set()
        tag = tk.simpledialog.askstring(parent=self.master, title='Add tag', prompt='Please input a tag name:') 
        
        self.tags_options_listboxs.insert(tk.END, tag)

    def delete_tag(self):
        self.tags_options_listboxs.delete(self.tags_options_listboxs.curselection())
        self.tags_options_listboxs.select_set(tk.END)

    def clear_interactive_window(self):
        self.canvas.delete(tk.ALL)
        self.controller.last_edge_scribbles = None

    def clear_show_logs(self):
        pass
    
    def switch_window(self):
        pass
    
    def draw_scribble(self, event):
        self.cnt += 1
        if self.controller.scribble_enable:
            
            if self.image_on_canvas.last_points is None:
                self.image_on_canvas.last_points = [event.x, event.y]
                return

            self.canvas.create_line(self.image_on_canvas.last_points[0], self.image_on_canvas.last_points[1], event.x, event.y, width=self.state['scribble_width'].get(), fill='yellow', tags=['es', 'es'+str(self.controller.interaction_round+1)])
            self.controller.edge_scribbles_draw.line(((self.image_on_canvas.last_points[0], self.image_on_canvas.last_points[1]), (event.x, event.y)), (255), width=1)
            self.image_on_canvas.last_points = [event.x, event.y]

            # FOR DEBUG
            # for i in self.canvas.find_all():
            #     print(f'id:{i} tags:{self.canvas.itemcget(i, "tags")}')
            # print(''.center(20, '='))

    def app_finish_object(self):
        self.canvas.delete('es')
        self.controller.finish_object()

    def app_undo_interaction(self):
        self.canvas.delete('es'+str(self.controller.interaction_round))
        self.controller.interaction_round -= 1
        self.controller.undo_click()

    def _load_image_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):
            filename = filedialog.askopenfilename(parent=self.master, filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*"),
            ], title="Choose an image")

            if len(filename) > 0:
                self.canvas.delete(tk.ALL)
                image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
                self.controller.set_image(image)

                self.dirs.delete(0,tk.END)
                self.dirs.insert(tk.END,os.path.basename(filename))

    def _load_directory_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):
            dir = filedialog.askdirectory(parent=self.master, title="Choose a directory")
            if len(dir) > 0:
                self.tdir = dir
                self.canvas.delete(tk.ALL)
                dirlist = os.listdir(self.tdir)
                dirlist.sort()
                self.dirs.delete(0,tk.END)
                
                for eachFile in dirlist:
                    self.dirs.insert(tk.END,eachFile)
                    self.dirs.config(selectbackground='LightSkyBlue')

    def _close_directory_callback(self):
        self.dirs.delete(0,tk.END)


    def select_image_from_dir_callback(self, *args):
        image_path = os.path.join(self.tdir, self.dirs.get(self.dirs.curselection()[0]))
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.canvas.delete(tk.ALL)
        self.controller.set_image(image)

    def _save_mask_callback(self):
        
        if self._check_entry(self):
            mask = self.controller.result_mask
            if mask is None:
                return

            filename = filedialog.asksaveasfilename(parent=self.master, initialfile='mask.png', filetypes=[
                ("PNG image", "*.png"),
                ("BMP image", "*.bmp"),
                ("All files", "*.*"),
            ], title="Save the mask as...")

            if len(filename) > 0:
                if mask.max() < 256:
                    mask = mask.astype(np.uint8)
                    mask *= 255 // mask.max()
                cv2.imwrite(filename, mask)

    def _load_mask_callback(self):
        if not self.controller.net.with_prev_mask:
            messagebox.showwarning("Warning", "The current model doesn't support loading external masks. "
                                              "Please use ITER-M models for that purpose.")
            return

        self.menubar.focus_set()
        if self._check_entry(self):
            filename = filedialog.askopenfilename(parent=self.master, filetypes=[
                ("Binary mask (png, bmp)", "*.png *.bmp"),
                ("All files", "*.*"),
            ], title="Chose an image")

            if len(filename) > 0:
                mask = cv2.imread(filename)[:, :, 0] > 127
                self.controller.set_mask(mask)
                self._update_image()

    def _about_callback(self):
        self.menubar.focus_set()

        text = [
            "Developed by:",
            "Wang Li",
            "The MIT License, 2022"
        ]

        messagebox.showinfo("About", '\n'.join(text))

    def _reset_last_object(self):
        self.state['alpha_blend'].set(0.5)
        self.state['prob_thresh'].set(0.5)
        self.controller.reset_last_object()
        self.canvas.delete('es')
        self.controller.edge_scribbles = self.controller.empty_mask.copy()

    def _update_prob_thresh(self, value):
        if self.controller.is_incomplete_mask:
            self.controller.prob_thresh = self.state['prob_thresh'].get()
            self._update_image()

    def _update_blend_alpha(self, value):
        self._update_image()

    def _update_click_radius(self, *args):
        if self.image_on_canvas is None:
            return

        self._update_image()

    def _update_scribble_linewidth(self, *args):
        # to do
        pass

    def _reset_predictor(self, *args, **kwargs):
        brs_mode = self.state['brs_mode'].get()
        prob_thresh = self.state['prob_thresh'].get()
        net_clicks_limit = None if brs_mode == 'NoBRS' else self.state['predictor_params']['net_clicks_limit'].get()

        if self.state['zoomin_params']['use_zoom_in'].get():
            zoomin_params = {
                'skip_clicks': self.state['zoomin_params']['skip_clicks'].get(),
                'target_size': self.state['zoomin_params']['target_size'].get(),
                'expansion_ratio': self.state['zoomin_params']['expansion_ratio'].get()
            }
            if self.state['zoomin_params']['fixed_crop'].get():
                zoomin_params['target_size'] = (zoomin_params['target_size'], zoomin_params['target_size'])
        else:
            zoomin_params = None

        predictor_params = {
            'brs_mode': brs_mode,
            'prob_thresh': prob_thresh,
            'zoom_in_params': zoomin_params,
            'predictor_params': {
                'net_clicks_limit': net_clicks_limit,
                'max_size': self.limit_longest_size
            },
            'brs_opt_func_params': {'min_iou_diff': 1e-3},
            'lbfgs_params': {'maxfun': self.state['lbfgs_max_iters'].get()}
        }
        self.controller.reset_predictor(predictor_params)

    def _click_callback(self, is_positive, x, y):
        self.canvas.focus_set()

        if self.image_on_canvas is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        if self._check_entry(self):
            self.controller.add_click(x, y, is_positive)

    def _update_image(self, reset_canvas=False):
        image = self.controller.get_visualization(alpha_blend=self.state['alpha_blend'].get(),
                                                  click_radius=self.state['click_radius'].get())
        if self.image_on_canvas is None:
            self.image_on_canvas = CanvasImage(self.canvas_frame, self.canvas)
            self.image_on_canvas.register_click_callback(self._click_callback)

        self._set_click_dependent_widgets_state()
        if image is not None:
            self.image_on_canvas.reload_image(Image.fromarray(image), reset_canvas)

    def _set_click_dependent_widgets_state(self):
        after_1st_click_state = tk.NORMAL if self.controller.is_incomplete_mask else tk.DISABLED
        before_1st_click_state = tk.DISABLED if self.controller.is_incomplete_mask else tk.NORMAL

        self.finish_object_button.configure(state=after_1st_click_state)
        self.undo_click_button.configure(state=after_1st_click_state)
        self.reset_clicks_button.configure(state=after_1st_click_state)
        self.zoomin_options_frame.set_frame_state(before_1st_click_state)
        

    def _check_entry(self, widget):
        all_checked = True
        if widget.winfo_children is not None:
            for w in widget.winfo_children():
                all_checked = all_checked and self._check_entry(w)

        if getattr(widget, "_check_bounds", None) is not None:
            all_checked = all_checked and widget._check_bounds(widget.get(), '-1')

        return all_checked
