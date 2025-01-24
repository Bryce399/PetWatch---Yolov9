import tkinter as tk

# 导入 detect_try.py 中的 detect_pet 函数
from detect_try import detect_pet



# 创建 tkinter 窗口
window = tk.Tk()
window.title("PetWatch")
window.geometry("600x400")

frame1=tk.Frame(window)
frame2=tk.Frame(window)
frame3=tk.Frame(window)

Pet=["cat","dog","bird"]
def set_pet_name():
    # 获取输入框中的宠物名称
    pet_name = entry.get()
    # 隐藏当前帧（页面）
    frame1.pack_forget()
    if pet_name in Pet:
        frame2.pack()
        detect_pet(pet_name)

    else:
        frame3.pack()


#初始界面
tk.Label(frame1, text="请输入您的宠物名称：").pack()
entry = tk.Entry(frame1)
entry.pack()
button = tk.Button(frame1, text="确定", command=set_pet_name)
button.pack()

#监控界面
tk.Label(frame2, text="").pack()

#界面三
tk.Label(frame3,text="抱歉，该系统不支持该宠物").pack()

frame1.pack()

# 进入消息循环
window.mainloop()