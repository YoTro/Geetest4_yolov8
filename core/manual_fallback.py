"""人工降级模块 - 提供 GUI 和 CLI 两种手动输入方式"""
import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from typing import List, Tuple
import logging

from utils import coordinate_utils

# --- GUI 相关函数 ---

def _run_gui_in_thread(
    main_image: Image.Image,
    ques_images: List[Image.Image],
    result_container: dict
):
    """在单独的线程中运行 Tkinter GUI 以支持超时。 (私有函数)"""
    root = None
    try:
        root = tk.Tk()
        root.title("人工验证")
        
        # Display question images
        if ques_images:
            top_frame = ttk.Frame(root, padding=5)
            top_frame.pack()
            for i, img in enumerate(ques_images):
                tk_img = ImageTk.PhotoImage(img)
                label = ttk.Label(top_frame, image=tk_img)
                label.image = tk_img 
                label.grid(row=0, column=i, padx=2)
        
        # Display main captcha image on a canvas
        canvas = tk.Canvas(root, width=main_image.width, height=main_image.height)
        canvas.pack()
        main_tk_img = ImageTk.PhotoImage(main_image)
        canvas.create_image(0, 0, anchor=tk.NW, image=main_tk_img)

        click_points = []
        start_time = time.time()

        def on_canvas_click(event):
            x, y = event.x, event.y
            click_points.append((x, y))
            canvas.create_oval(x-3, y-3, x+3, y+3, fill="red", outline="red")
        
        canvas.bind("<Button-1>", on_canvas_click)

        def on_submit():
            end_time = time.time()
            passtime = int((end_time - start_time) * 1000)
            # The GUI method has access to image dimensions, so it can do the conversion
            geetest_coords = coordinate_utils.convert_to_geetest_format(
                click_points, container_size=(main_image.width, main_image.height)
            )
            result_container['result'] = (geetest_coords, passtime)
            root.quit()
            root.destroy()

        ttk.Button(root, text="提交", command=on_submit).pack(pady=5)
        root.mainloop()

    except Exception as e:
        # This will catch Tkinter errors on headless systems
        logging.getLogger(__name__).error(f"GUI 运行失败 (可能是无头系统): {e}", exc_info=True)
        if root:
            root.quit()
            root.destroy()
        result_container['result'] = (None, 0) # Return None to signal failure

def get_user_input_gui(
    main_image: Image.Image,
    ques_images: List[Image.Image],
    timeout: int = 60
) -> Tuple[List[List[int]], int]:
    """
    通过 GUI 获取用户输入。返回已经转换的geetest格式坐标。
    """
    logger = logging.getLogger(__name__)
    result_container = {}
    gui_thread = threading.Thread(
        target=_run_gui_in_thread,
        args=(main_image, ques_images, result_container),
        daemon=True
    )
    gui_thread.start()
    gui_thread.join(timeout=timeout)

    if 'result' in result_container and result_container['result'][0] is not None:
        return result_container['result']
    
    logger.warning("用户输入超时或GUI初始化失败。")
    return [], 0


# --- CLI 相关函数 ---

def get_user_input_cli(
    num_points: int
) -> Tuple[List[Tuple[int, int]], int]:
    """
    通过 CLI 获取用户输入。返回原始像素坐标。
    """
    click_points = []
    
    print("\n" + "="*60)
    print("=== 进入命令行手动验证模式 ===")
    print("\n请按照目标文字顺序，依次输入您在主图上点击的 x,y 坐标。\n")
    print("格式为 'x,y' (例如: 123,45)。 x:(0, 300), y(0, 200)\n每输入一个坐标后按 Enter。")
    print("="*60)
    
    start_time = time.time()
    
    for i in range(num_points):
        try:
            raw_input = input(f"请输入第 {i+1} 个坐标 (x,y): ").strip()
            parts = raw_input.replace(' ', '').split(',')
            if len(parts) != 2:
                raise ValueError("格式错误，必须是 'x,y'。")
                
            x = int(parts[0])
            y = int(parts[1])
            click_points.append((x, y))

        except (ValueError, IndexError) as e:
            logging.getLogger(__name__).error(f"无效输入: {e}。手动验证失败。")
            return [], 0
        except Exception as e:
            logging.getLogger(__name__).error(f"未知输入错误: {e}。手动验证失败。")
            return [], 0
    
    end_time = time.time()
    passtime = int((end_time - start_time) * 1000)
    
    logging.getLogger(__name__).info(f"收到的原始像素坐标: {click_points}")
    
    return click_points, passtime
