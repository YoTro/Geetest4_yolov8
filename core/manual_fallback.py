"""
人工降级模块 - 使用Tkinter提供GUI界面
"""
import time
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from typing import List, Optional, Tuple
import requests
import logging

from utils import image_processor, coordinate_utils

def _run_gui_in_thread(
    main_image: Image.Image,
    ques_images: List[Image.Image],
    result_container: dict
):
    root = None
    try:
        root = tk.Tk()
        root.title("人工验证")
        
        if ques_images:
            top_frame = ttk.Frame(root, padding=5)
            top_frame.pack()
            current_x = 0
            for img in ques_images:
                tk_img = ImageTk.PhotoImage(img)
                label = ttk.Label(top_frame, image=tk_img)
                label.image = tk_img
                label.grid(row=0, column=current_x, padx=2)
                current_x += 1
        
        canvas = tk.Canvas(root, width=main_image.width, height=main_image.height)
        canvas.pack()
        main_tk_img = ImageTk.PhotoImage(main_image)
        canvas.create_image(0, 0, anchor=tk.NW, image=main_tk_img)
        canvas.image = main_tk_img

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
            geetest_coords = coordinate_utils.convert_to_geetest_format(
                click_points, container_size=(main_image.width, main_image.height)
            )
            result_container['result'] = (geetest_coords, passtime)
            root.quit()
            root.destroy()

        ttk.Button(root, text="提交", command=on_submit).pack(pady=5)
        root.mainloop()

    except Exception as e:
        logging.getLogger(__name__).error(f"GUI 运行失败: {e}", exc_info=True)
        if root:
            root.quit()
            root.destroy()
        result_container['result'] = ([], 0)

def get_user_input_with_gui(
    main_image_url: str,
    ques_image_urls: List[str],
    session: requests.Session,
    timeout: int = 60
) -> Tuple[List[List[int]], int]:
    logger = logging.getLogger(__name__)

    main_image = image_processor.download_image(session, main_image_url)
    if not main_image:
        logger.error("无法下载主验证码图片。")
        return [], 0
    
    ques_images = [img for url in ques_image_urls if (img := image_processor.download_image(session, url))]

    result_container = {}
    gui_thread = threading.Thread(
        target=_run_gui_in_thread,
        args=(main_image, ques_images, result_container),
        daemon=True
    )
    gui_thread.start()
    gui_thread.join(timeout=timeout)

    if 'result' in result_container:
        return result_container['result']
    
    logger.warning("用户输入超时。")
    return [], 0
