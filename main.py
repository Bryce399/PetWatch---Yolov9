from UI.main_p import Main_P
from UI.navigator import Navigator
from UI.analog import Analog
from UI.Realtime import Realtime
from UI.warn import Warn
from UI.diary import Diary
from UI.analyze import Analyze
from UI.evday import Evday
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import STM
import argparse
import os
import platform
import sys
from pathlib import Path
import jieba
from datetime import datetime,date,timedelta
import torch
import json

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets

class Controller:
    def __init__(self):
        pass
    def show_login(self):
        self.login = navigator()
        self.login.switch_window.connect(self.show_main)
        self.login.show()

    def show_main(self,text):
        self.Pet = ["cat", "dog", "bird"]
        global pet
        pet=text
        if text in self.Pet:
            self.window = main_p()
            self.login.close()
            self.window.show()
        else:
            self.window = Wrong(text)
            self.window.show()
class Wrong(QtWidgets.QWidget):
    switch_window = QtCore.pyqtSignal(str)

    def __init__(self,text):
        QtWidgets.QWidget.__init__(self)
        self.setWindowTitle('PetWatch')
        self.setGeometry(1000, 500, 300, 100)
        layout = QtWidgets.QGridLayout()
        self.label2 = QtWidgets.QLabel("抱歉，该系统不支持该宠物")
        self.label3=QtWidgets.QLabel("注：该系统仅支持：cat,dog,bird")
        layout.addWidget(self.label2)
        layout.addWidget(self.label3)

        self.setLayout(layout)
class navigator(Navigator,QtWidgets.QMainWindow):
    switch_window = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super(navigator, self).__init__(parent)
        self.setupUi(self)
        self.background()
        global BQ
        BQ=False

    def background(self):
        self.pushButton.clicked.connect(self.login)

    def login(self):
        self.switch_window.emit(self.lineEdit.text())
class main_p(Main_P,QtWidgets.QMainWindow):
    switch_window = QtCore.pyqtSignal(str)

    def __init__(self,parent=None):
        super(main_p, self).__init__(parent)
        self.setupUi(self)
        self.background()


    def background(self):
        self.pushButton.clicked.connect(self.open_1)
        self.pushButton_2.clicked.connect(self.open_2)
        self.pushButton_3.clicked.connect(self.open_3)
        self.pushButton_4.clicked.connect(self.open_4)

    def open_1(self):
        self.close()
        self.realtime=realtime()
        self.realtime.show()
        f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "打开实时监控系统" + "\n")

    def open_2(self):
        self.close()
        self.analog=analog()
        self.analog.show()
        f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "打开模拟检测系统" + "\n")

    def open_3(self):
        self.close()
        self.warn=warn()
        self.warn.show()

    def open_4(self):
        self.close()
        self.diray=diray()
        self.diray.show()
class realtime(Realtime,QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super(realtime, self).__init__(parent)
        self.setupUi(self)
        self.background()

        sys.stdout = EmittingStr(textWritten=self.outputWritten)
        sys.stderr = EmittingStr(textWritten=self.outputWritten)

    def background(self):
        self.lineEdit.setText(pet)
        date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.lineEdit_2.setText(date_time)
        self.pushButton.clicked.connect(self.parse_1)
        self.pushButton_2.clicked.connect(self.close_1)
        self.pushButton_3.clicked.connect(self.back)
        self.pushButton_4.clicked.connect(self.exit)

    def parse_1(self):
        global End
        End = False
        global CS, CB
        CS, CB = 0, 0
        self.textEdit.clear()
        self.label_2.clear()
        self.pushButton_3.setEnabled(False)
        self.label_3.setPixmap(QtGui.QPixmap("safe.png"))
        global WA
        WA=True
        print("打开摄像头")
        f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "打开摄像头" + "\n")
        opt = self.parse_opt()
        self.main(opt)

    def close_1(self):
        print("关闭摄像头")
        f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "关闭摄像头" + "\n")
        global End
        End = True
        self.pushButton_3.setEnabled(True)
        self.label_2.clear()

    def main(self,opt):
        check_requirements(exclude=('tensorboard', 'thop'))
        self.run(**vars(opt))

    def parse_opt(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp/weight/best.pt',
                            help='model path or triton URL')
        parser.add_argument('--source', type=str, default=ROOT / "0", help='file/dir/URL/glob/screen/0(webcam)')
        parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[320],
                            help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
        print_args(vars(opt))
        return opt

    def display_image(self, image_data):
        image_data=cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        height, width, channel = image_data.shape
        bytes_per_line = channel * width
        qimage = QImage(image_data.data, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimage)
        self.label_2.setPixmap(pixmap)
        self.label_2.repaint()

    def run(self,
            weights=ROOT / 'yolo.pt',  # model path or triton URL
            source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
            data=ROOT / 'data/coco.yaml',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            vid_stride=1,  # video frame-rate stride
    ):
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        if webcam:
            view_img = check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            view_img = True
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

        today_anly = str(date.today())
        f_anly_name = "diary_al\\" + today_anly + ".txt"
        f_anly = open(f_anly_name, "a")

        DAN_1 = []
        DAN_2 = []
        DAN_3 = []
        DAN_4 = []
        DAN_5 = []
        DAN = []
        for path, im, im0s, vid_cap, s in dataset:
            Dan_1,Dan_2,Dan_3,Dan_4=0,0,0,0
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = pred[0][1] if isinstance(pred[0], list) else pred[0]
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Stream results
                im0 = annotator.result()
                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    im1 = cv2.resize(im0, (641, 471))
                    self.display_image(im1)


                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
            # print(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
            lst = jieba.lcut(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms", cut_all=True)

            global CS, pet, CB
            if CS <= 18:
                if pet in lst:
                    if CB == 0:
                        print("检测到宠物")
                        f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "检测到宠物" + "\n")
                        CB = 1
                    CS = 0
                else:
                    CS = CS + 1
            else:
                if CB == 0:
                    print("未检测到宠物")
                    f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "未检测到宠物" + "\n")
                    CS = 0
                if CB == 1:
                    print("宠物丢失")
                    f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "宠物丢失" + "\n")
                    self.label_3.setPixmap(QtGui.QPixmap("danger.png"))
                    CS = 0
                    global WA
                    if WA==True:
                        print("发送邮箱")
                        self.content = """PetWatch提醒您，您的宠物"""+pet_name+"""已经消失在监控范围内，请您注意您的宠物是否丢失。"""
                        send_em(qq_1, self.content)
                        WA = False


            Dan_5 = datetime.now().strftime('%H:%M:%S')
            DAN_5.append(Dan_5)
            if "wine" in lst:
                Dan_1 = Dan_1 + 1
            if "cup" in lst:
                Dan_1 = Dan_1 + 1
            if "bowl" in lst:
                Dan_1 = Dan_1 + 1
            if "bottle" in lst:
                Dan_1 = Dan_1 + 1
            DAN_1.append(Dan_1)
            if Dan_1 > 0:
                print(Dan_5)
                print("注意易碎品，易碎品被打破可能会对宠物造成伤害")
                self.label_3.setPixmap(QtGui.QPixmap("danger.png"))
                f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "周围存在易碎品" + "\n")

            if "fork" in lst:
                Dan_2 = Dan_2 + 1
            if "knife" in lst:
                Dan_2 = Dan_2 + 1
            if "spoon" in lst:
                Dan_2 = Dan_2 + 1
            DAN_2.append(Dan_2)
            if Dan_2 > 0:
                print(Dan_5)
                print("注意锋利物品，可能会对宠物造成伤害")
                self.label_3.setPixmap(QtGui.QPixmap("danger.png"))
                f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "周围存在锋利物品" + "\n")

            if "microwave" in lst:
                Dan_3 = Dan_3 + 1
            if "oven" in lst:
                Dan_3 = Dan_3 + 1
            if "toaster" in lst:
                Dan_3 = Dan_3 + 1
            DAN_3.append(Dan_3)
            if Dan_3 > 0:
                print(Dan_5)
                print("电器可能导致宠物电击或烫伤")
                self.label_3.setPixmap(QtGui.QPixmap("danger.png"))
                f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "周围存在电器" + "\n")

            if "tv" in lst:
                Dan_4 = Dan_4 + 1
            if "laptop" in lst:
                Dan_4 = Dan_4 + 1
            if "cell phone" in lst:
                Dan_4 = Dan_4 + 1
            DAN_4.append(Dan_4)
            if Dan_4 > 0:
                print(Dan_5)
                print("小心电子产品损害")
                self.label_3.setPixmap(QtGui.QPixmap("danger.png"))
                f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "周围存在贵重电子物品" + "\n")

            if End == True:
                self.label_2.clear()
                break

        DAN.append(DAN_5)
        DAN.append(DAN_1)
        DAN.append(DAN_2)
        DAN.append(DAN_3)
        DAN.append(DAN_4)
        DAN.append(sum(DAN_1) + sum(DAN_2) + sum(DAN_3) + sum(DAN_4))
        json.dump(DAN, f_anly)
        f_anly.write("\n")
        #print(DAN)
        f_anly.close()

        #DAN.append(DAN_5,DAN_1,DAN_2,DAN_3,DAN_4)
        #json.dump(self.DAN, f_anly)
        # print(self.DAN)


        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


    def outputWritten(self, text):
        cursor = self.textEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textEdit.setTextCursor(cursor)
        self.textEdit.ensureCursorVisible()
    def back(self):
        self.close()
        self.main_p=main_p()
        self.main_p.show()
        f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "返回主页" + "\n")

    def exit(self):
        f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "退出PetWatch系统" + "\n")
        sys.exit(app.exec_())
class analog(Analog,QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super(analog, self).__init__(parent)
        self.setupUi(self)
        self.background()

        sys.stdout = EmittingStr(textWritten=self.outputWritten)
        sys.stderr = EmittingStr(textWritten=self.outputWritten)

    def background(self):
        self.lineEdit.setText(pet)
        self.pushButton.clicked.connect(self.photo)
        self.pushButton_2.clicked.connect(self.video)
        self.pushButton_3.clicked.connect(self.back)
        self.pushButton_4.clicked.connect(self.exit)

    def photo(self):
        global CS, CB
        CS, CB = 0, 0
        global WA
        WA = True
        self.label_2.clear()
        self.textEdit.clear()
        self.label_3.setPixmap(QtGui.QPixmap("safe.png"))
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)
        self.Path_1 = self.button_photo_open()
        if self.Path_1==False:
            print("未打开文件")
            self.pushButton_2.setEnabled(True)
            self.pushButton_3.setEnabled(True)
        else:
            print("图片检测")
            opt = self.parse_opt(self.Path_1)
            self.main(opt)
            print("图片检测完成")
            self.pushButton_2.setEnabled(True)
            self.pushButton_3.setEnabled(True)

    def video(self):
        global CS, CB
        CS,CB= 0,0
        global WA
        WA=True
        self.label_3.setPixmap(QtGui.QPixmap("safe.png"))
        self.label_2.clear()
        self.textEdit.clear()
        #self.pushButton_1.setEnabled(False)
        #self.pushButton_3.setEnabled(False)
        self.Path_1 = self.button_video_open()
        if self.Path_1 == False:
            print("未打开文件")
        else:
            print("视频检测")
            print("注意该过程不可中断")
            f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "视频模拟检测" + "\n")
            opt = self.parse_opt(self.Path_1)
            self.main(opt)
            print("视频检测完成")
            f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "视频检测完成" + "\n")

    def button_photo_open(self):
        photo_name,_=QtWidgets.QFileDialog.getOpenFileName(None, "选择图片文件", "", "图片文件 (*.jpg *.jpeg *.png *.bmp *.gif)")
        if not photo_name:
            return False
        return photo_name

    def button_video_open(self):
        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(None, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)")
        if not video_name:
            return False
        return video_name

    def outputWritten(self, text):
        cursor = self.textEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textEdit.setTextCursor(cursor)
        self.textEdit.ensureCursorVisible()

    def display_image(self, image_data):
        image_data=cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        height, width, channel = image_data.shape
        bytes_per_line = channel * width
        qimage = QImage(image_data.data, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimage)
        self.label_2.setPixmap(pixmap)
        self.label_2.repaint()

    def main(self, opt):
        check_requirements(exclude=('tensorboard', 'thop'))
        self.run(**vars(opt))

    def parse_opt(self, Path_1):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp/weight/best.pt',
                            help='model path or triton URL')
        parser.add_argument('--source', type=str, default=ROOT / Path_1, help='file/dir/URL/glob/screen/0(webcam)')
        parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml',
                            help='(optional) dataset.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[320],
                            help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int,
                            help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
        print_args(vars(opt))
        return opt


    def run(self,
            weights=ROOT / 'yolo.pt',  # model path or triton URL
            source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
            data=ROOT / 'data/coco.yaml',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            vid_stride=1,  # video frame-rate stride
    ):
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        if webcam:
            view_img = check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            view_img = True
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

        today_anly = str(date.today())
        f_anly_name = "diary_al\\" + today_anly + ".txt"
        f_anly = open(f_anly_name, "a")

        self.DAN_1=[]
        self.DAN_2=[]
        self.DAN_3=[]
        self.DAN_4=[]
        self.DAN_5=[]
        self.DAN=[]
        for path, im, im0s, vid_cap, s in dataset:
            self.Dan_1=0
            self.Dan_2=0
            self.Dan_3=0
            self.Dan_4=0
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = pred[0][1] if isinstance(pred[0], list) else pred[0]
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Stream results
                im0 = annotator.result()
                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    im1 = cv2.resize(im0, (641, 471))
                    self.display_image(im1)


                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
            # print(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
            lst = jieba.lcut(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms", cut_all=True)

            global CS,pet,CB
            if CS <= 18:
                if pet in lst:
                    if CB == 0:
                        print("检测到宠物")
                        f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "检测到宠物" + "\n")
                        CB = 1
                    CS = 0
                else:
                    CS = CS + 1
            else:
                if CB == 0:
                    print("未检测到宠物")
                    f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "未检测到宠物" + "\n")
                    CS = 0
                if CB == 1:
                    print("宠物丢失")
                    f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "宠物丢失" + "\n")
                    self.label_3.setPixmap(QtGui.QPixmap("danger.png"))
                    CS = 0
                    CB = 0
                    global WA
                    if WA == True:
                        print("发送邮箱")
                        self.content = """PetWatch提醒您，您的宠物"""+pet_name+"""已经消失在监控范围内，请您注意您的宠物是否丢失。"""
                        send_em(qq_1, self.content)
                        WA = False



            self.Dan_5 = datetime.now().strftime('%H:%M:%S')
            self.DAN_5.append(self.Dan_5)

            if "wine" in lst:
                self.Dan_1 = self.Dan_1 + 1
            if "cup" in lst:
                self.Dan_1 = self.Dan_1 + 1
            if "bowl" in lst:
                self.Dan_1 = self.Dan_1 + 1
            if "bottle" in lst:
                self.Dan_1 = self.Dan_1 + 1
            self.DAN_1.append(self.Dan_1)
            if self.Dan_1>0:
                print(self.Dan_5)
                print("注意易碎品，易碎品被打破可能会对宠物造成伤害")
                self.label_3.setPixmap(QtGui.QPixmap("danger.png"))
                f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "周围存在易碎物品" + "\n")

            
            if "fork" in lst:
                self.Dan_2=self.Dan_2+1
            if "knife" in lst:
                self.Dan_2=self.Dan_2+1
            if "spoon" in lst:
                self.Dan_2=self.Dan_2+1
            self.DAN_2.append(self.Dan_2)
            if self.Dan_2>0:
                print(self.Dan_5)
                print("注意锋利物品，可能会对宠物造成伤害")
                self.label_3.setPixmap(QtGui.QPixmap("danger.png"))
                f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "周围存在锋利物品" + "\n")

            if "microwave" in lst:
                self.Dan_3=self.Dan_3+1
            if "oven" in lst:
                self.Dan_3=self.Dan_3+1
            if "toaster" in lst:
                self.Dan_3 = self.Dan_3 + 1
            self.DAN_3.append(self.Dan_3)
            if self.Dan_3>0:
                print(self.Dan_5)
                print("电器可能导致宠物电击或烫伤")
                self.label_3.setPixmap(QtGui.QPixmap("danger.png"))
                f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "周围存在电器" + "\n")

            if "tv" in lst:
                self.Dan_4=self.Dan_4+1
            if "laptop" in lst:
                self.Dan_4 =self.Dan_4 + 1
            if "cell phone" in lst:
                self.Dan_4 = self.Dan_4 + 1
            self.DAN_4.append(self.Dan_4)
            if self.Dan_4>0:
                print(self.Dan_5)
                print("小心电子产品损害")
                self.label_3.setPixmap(QtGui.QPixmap("danger.png"))
                f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "周围存在贵重电子物品" + "\n")


        self.DAN.append(self.DAN_5)
        self.DAN.append(self.DAN_1)
        self.DAN.append(self.DAN_2)
        self.DAN.append(self.DAN_3)
        self.DAN.append(self.DAN_4)
        self.DAN.append(sum(self.DAN_1)+sum(self.DAN_2)+sum(self.DAN_3)+sum(self.DAN_4))
        json.dump(self.DAN, f_anly)
        f_anly.write("\n")
        #print(self.DAN)
        f_anly.close()


        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    def back(self):
        self.close()
        self.main_p = main_p()
        self.main_p.show()
        f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "返回主界面" + "\n")

    def exit(self):
        f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "PetWacth系统关闭" + "\n")
        sys.exit(app.exec_())

class warn(Warn,QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super(warn, self).__init__(parent)
        self.setupUi(self)
        self.background()
        if BQ==True:
            self.lineEdit.setText(pet_name)
            self.lineEdit_2.setText(year_1)
            self.lineEdit_3.setText(month_1)
            self.lineEdit_4.setText(day_1)
            self.lineEdit_5.setText(name_1)
            self.lineEdit_6.setText(qq_1)
            self.lineEdit_7.setText(phone_1)


    def background(self):
        self.pushButton.clicked.connect(self.back)
        self.pushButton_2.clicked.connect(self.exit)
        self.pushButton_3.clicked.connect(self.Queding)

    def back(self):
        self.close()
        self.main_p = main_p()
        self.main_p.show()

    def exit(self):
        f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "PetWatch系统关闭" + "\n")
        sys.exit(app.exec_())

    def Queding(self):
        global pet_name,year_1,month_1,day_1,name_1,qq_1,phone_1
        pet_name=self.lineEdit.text()
        year_1=self.lineEdit_2.text()
        month_1=self.lineEdit_3.text()
        day_1=self.lineEdit_4.text()
        name_1=self.lineEdit_5.text()
        qq_1=self.lineEdit_6.text()
        phone_1=self.lineEdit_7.text()
        global BQ
        BQ=True
        self.content="""尊敬的用户"""+name_1+"""，PetWatch提醒您已经成功导入邮件信息，如果宠物遇到重大问题，将通过邮件的方式提醒您。"""
        send_em(qq_1,self.content)
        #print(year_1+"-"+month_1+"-"+day_1)
        #print(date.today())
        if year_1+"-"+month_1+"-"+day_1==str(date.today()):
            #print(True)
            self.content_2="""PetWatch祝您宠物"""+pet_name+"""今日生日快乐！。"""
            send_em(qq_1,self.content_2 )
class diray(Diary,QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super(diray, self).__init__(parent)
        self.setupUi(self)
        self.background()

    def background(self):
        self.pushButton.clicked.connect(self.back)
        self.pushButton_2.clicked.connect(self.exit)
        self.pushButton_3.clicked.connect(self.anal)
        self.pushButton_4.clicked.connect(self.evd)

    def anal(self):
        self.close()
        self.analyze = analyze()
        self.analyze.show()

    def evd(self):
        self.close()
        self.evday = evday()
        self.evday.show()
    def back(self):
        self.close()
        self.main_p = main_p()
        self.main_p.show()

    def exit(self):
        f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "PetWatch系统关闭" + "\n")
        sys.exit(app.exec_())
class analyze(Analyze,QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super(analyze, self).__init__(parent)
        self.setupUi(self)
        self.background()

    def background(self):
        self.pushButton.clicked.connect(self.analy)
        self.pushButton_2.clicked.connect(self.back)
        self.pushButton_3.clicked.connect(self.exit)

    def analy(self):
        self.year_2=self.lineEdit.text()
        self.month_2=self.lineEdit_2.text()
        self.day_2=self.lineEdit_3.text()
        self.year_3=self.lineEdit_4.text()
        self.month_3=self.lineEdit_5.text()
        self.day_3=self.lineEdit_6.text()
        self.open_file()


    def open_file(self):
        start_date =self.year_2+"-"+self.month_2+"-"+self.day_2
        end_date = self.year_3+"-"+self.month_3+"-"+self.day_3
        current_date = start_date
        Te=[]
        Time = []
        Thing = []

        while current_date <= end_date:
            file_path_a = "diary_al\\"+current_date + ".txt"
            #print(file_path_a)

            if os.path.exists(file_path_a):
                T_A=self.read_a_time(file_path_a)
                T_A=int(T_A)
                Time.append(T_A)
                Te.append(current_date[-5:])
                T_H=self.read_a_thing(file_path_a)
                if T_A!=0:
                    T_H=T_H/int(T_A)
                Thing.append(T_H)

            else:
                Time.append(0)
                Te.append(current_date[-5:])
                Thing.append(0)

            if current_date > end_date:
                break

            current_date = (datetime.strptime(current_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

        self.fig_1 = Figure()
        self.fig_1.clear()
        ax_1 = self.fig_1.add_subplot(111)
        ax_1.plot(Te,Time)
        pixmap_1 = self.fig_to_pixmap(self.fig_1)
        self.label_2.setPixmap(pixmap_1)

        self.fig_2 = Figure()
        self.fig_2.clear()
        ax_2 = self.fig_2.add_subplot(111)
        ax_2.plot(Te, Thing)
        pixmap_2 = self.fig_to_pixmap(self.fig_2)
        self.label_3.setPixmap(pixmap_2)

    def fig_to_pixmap(self,fig):
        canvas = FigureCanvas(fig)
        canvas.setFixedSize(451, 441)
        canvas.draw()
        width, height = canvas.get_width_height()
        pixmap = canvas.grab()
        return pixmap

    def back(self):
        self.close()
        self.main_p = main_p()
        self.main_p.show()

    def exit(self):
        f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "PetWatch系统关闭" + "\n")
        sys.exit(app.exec_())

    def read_a_time(self,file_path_a):
        f_a_anly = open(file_path_a, "r")
        file_ar=f_a_anly.readlines()
        time_a = 0
        for fa in file_ar:
            fa_1 = fa.strip().find("]")
            fail_a = fa.strip()[2:fa_1]
            t1_sr = fail_a[1:9]
            t2_sr = fail_a[-9:-1]
            if int(t1_sr[:1]) <= int(t2_sr[:1]):
                t1 = int(t1_sr[:2]) * 60 + int(t1_sr[3:5]) + int(t1_sr[-2:]) / 60
                t2 = int(t2_sr[:2]) * 60 + int(t2_sr[3:5]) + int(t2_sr[-2:]) / 60
            else:
                t1 = int(t1_sr[:2]) * 60 + int(t1_sr[3:5]) + int(t1_sr[-2:]) / 60
                t2 = 24 * 60
            Time_r = int(t2 - t1)
            time_a = time_a + Time_r
        return time_a

    def read_a_thing(self,file_path_a):
        f_a_anly = open(file_path_a, "r")
        file_ar = f_a_anly.readlines()
        TH_a=0
        for fa in file_ar:
            Th_a=int(fa.strip().split(",")[-1][:-1])
            TH_a=Th_a+TH_a
        return TH_a
class evday(Evday,QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super(evday, self).__init__(parent)
        self.setupUi(self)
        self.background()

        sys.stdout = EmittingStr(textWritten=self.outputWritten)
        sys.stderr = EmittingStr(textWritten=self.outputWritten)

    def background(self):
        self.pushButton.clicked.connect(self.evda)
        self.pushButton_2.clicked.connect(self.back)
        self.pushButton_3.clicked.connect(self.exit)

    def outputWritten(self, text):
        cursor = self.textEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textEdit.setTextCursor(cursor)
        self.textEdit.ensureCursorVisible()

    def evda(self):
        self.textEdit.clear()
        self.year_4=str(self.lineEdit.text())
        self.month_4=str(self.lineEdit_2.text())
        self.day_4=str(self.lineEdit_3.text())
        f_day_name="diary\\"+self.year_4+"-"+self.month_4+"-"+self.day_4+".txt"
        f_day=open(f_day_name,'r')
        f_day_r=f_day.readlines()
        for i in f_day_r:
            print(i)
        f_day.close()

    def back(self):
        self.close()
        self.main_p = main_p()
        self.main_p.show()

    def exit(self):
        f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "PetWatch系统关闭" + "\n")
        sys.exit(app.exec_())
class EmittingStr(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(0, loop.quit)
        loop.exec_()
def send_em(qq_1,content):
    email = MIMEMultipart()
    email['From'] = '18098731679@163.com'
    email['To'] = qq_1
    email['Subject'] = Header('PetWatch提醒', 'utf-8')

    content = content
    email.attach(MIMEText(content, 'plain', 'utf-8'))

    smtp_obj = STM.SMTP_SSL('smtp.163.com', 465)

    smtp_obj.login('18098731679@163.com', 'VBQQBUSOOVAVJCRX')

    smtp_obj.sendmail(
        '18098731679@163.com',
        [qq_1],
        email.as_string()
    )

qq_1="13098731679@qq.com"

if __name__=="__main__":
    today_pet = str(date.today())
    f_r_name = "diary\\" + today_pet + ".txt"
    f_r = open(f_r_name, "a")
    f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "PetWatch系统启动" + "\n")

    app = QApplication(sys.argv)
    controller = Controller()
    controller.show_login()
    sys.exit(app.exec_())

    f_r.write(str(datetime.now().strftime("%H:%M:%S")) + ":" + "PetWatch系统关闭" + "\n")
    f_r.close()