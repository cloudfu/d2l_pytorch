# coding:utf-8
import sys
import time
from hashlib import sha1
import requests
from texttable import Texttable

# 说明：飞鹅云后台配置参数
URL = "http://api.feieyun.cn/Api/Open/"  # 不需要修改
USER = "55387938@qq.com"  # *必填*：飞鹅云后台注册账号
UKEY = "HPJtdVP3FZSBnkJ2"  # *必填*: 飞鹅云后台注册账号后生成的UKEY 【备注：这不是填打印机的KEY】
SN = "960800945"  # *必填*：打印机编号，必须要在管理后台里手动添加打印机或者通过API添加之后，才能调用API


# 飞鹅云后台入口和账号
# https://admin.feieyun.com/index.php
# 55387938@qq.com
# 12345678


class UserVisionData:

    def __init__(self, args):
        """
        打印数据初始化
        :param args:Excel导出参数，通过，进行分割
        """
        # 字符串分割
        array_print_data = args.split(",")

        # -- 验光信息 --
        self.SPH_RightEye = ""
        self.SPH_LeftEye = ""

        # 散光（CYL）
        self.CYL_RightEye = ""
        self.CYL_LeftEye = ""
        
        # 轴位（AXIS）
        self.AXIS_RightEye = ""
        self.AXIS_LeftEye = ""

        # 下加光
        self.DOWN_RightEye = ""
        self.DOWN_LeftEye = ""

        # 瞳距
        self.DIST_RightEye = ""
        self.DIST_LeftEye = ""

        # 判断字符长度
        if len(array_print_data) == 21:
            self.title = "5.76眼镜臻选"
            self.printDateTime = time.strftime("%Y-%m-%d %H:%M:%S")

            # -- 用户信息 --
            self.userName = array_print_data[0]
            self.phoneNum = array_print_data[1]
            self.billOwner = array_print_data[2]
            self.Purpose = array_print_data[3]

            # -- 验光信息 --
            # 度数（SPH）
            if array_print_data[4].strip() != "":
                self.SPH_RightEye = '%+.2f' % float(array_print_data[4])
            if array_print_data[5].strip() != "":
                self.SPH_LeftEye = '%+.2f' % float(array_print_data[5])
            # 散光（CYL）
            if array_print_data[6].strip() != "":
                self.CYL_RightEye = '%+.2f' % float(array_print_data[6])
            if array_print_data[7].strip() != "":
                self.CYL_LeftEye = '%+.2f' % float(array_print_data[7])
            # 轴位（AXIS）
            if array_print_data[8].strip() != "":
                self.AXIS_RightEye = array_print_data[8]
            if array_print_data[9].strip() != "":
                self.AXIS_LeftEye = array_print_data[9]
            # 下加光
            if array_print_data[10].strip() != "":
                self.DOWN_RightEye = '%+.2f' % float(array_print_data[10])
            if array_print_data[11].strip() != "":
                self.DOWN_LeftEye = '%+.2f' % float(array_print_data[11])
            # 瞳距
            if array_print_data[12].strip() != "":
                self.DIST_RightEye = '%.1f' % float(array_print_data[12])
            if array_print_data[13].strip() != "":
                self.DIST_LeftEye = '%.1f' % float(array_print_data[13])

            # -- 商品信息 --
            self.GlassName = array_print_data[14]
            self.GlassType = array_print_data[15]
            self.GlassFrameName = array_print_data[16]
            self.GlassFrameMode = array_print_data[17]
            self.Comments = array_print_data[18]

            # 打印配置参数
            self.filePath = array_print_data[19]
            self.printType = array_print_data[20]

            # 需要打印内容记录
            self.logcat(args)

    def printTextTable(self):
        """
        打印表格数据
        :return:
        """
        table_user_info = Texttable()
        table_user_info.set_deco(Texttable.HEADER)
        table_user_info.set_header_align(["l", "l"])
        table_user_info.set_cols_align(["r", "l"])
        table_user_info.add_rows([["用户信息", ""],
                                  ["姓名:", self.userName, ],
                                  ["手机号:", self.phoneNum],
                                  ["所属人:", self.billOwner],
                                  ["配镜用途:", self.Purpose],
                                  ["打印时间:", self.printDateTime],
                                  ["备注:", self.Comments]])
        str_user_info = table_user_info.draw()
        print(str_user_info)
        # self.logcat(str_user_info)

        print()

        table_eyes_info = Texttable()
        table_eyes_info.set_deco(Texttable.HEADER)
        table_eyes_info.set_header_align(["c", "c", "c", "c", "c", "c"])
        table_eyes_info.set_cols_align(["c", "c", "c", "c", "c", "c"])
        table_eyes_info.add_rows([["", "度数", "散光", "轴位", "下加光", "瞳距"],
                                  ["右眼:", self.SPH_RightEye, self.CYL_RightEye, self.AXIS_RightEye, self.DOWN_RightEye,
                                   self.DIST_RightEye],
                                  ["左眼:", self.SPH_LeftEye, self.CYL_LeftEye, self.AXIS_LeftEye, self.DOWN_LeftEye,
                                   self.DIST_LeftEye]])
        str_eyes_info = table_eyes_info.draw()
        print(str_eyes_info)
        # self.logcat(str_eyes_info)

    def getHtmlData(self):
        """
        获取云端打印数据格式
        :return:返回html数据打印格式
        """

        split_line = "------------------------------------------"

        # 用户基本信息
        html = "<CB>" + self.title + "</CB>\n"
        html += "<CB>您眼睛BUG的修复师</CB>\n"
        html += "姓名：" + self.userName + "\n"
        html += "电话号码：" + self.phoneNum + "\n"
        html += "所 属 人：" + self.billOwner + "\n"
        html += "配镜用途：" + self.Purpose + "\n"
        html += split_line + "\n"
        html += "验光数据：\n"

        # 验光信息
        table_eyes_info = Texttable()
        table_eyes_info.set_deco(Texttable.HEADER)
        table_eyes_info.set_header_align(["c", "c", "c", "c", "c", "c"])
        table_eyes_info.set_cols_align(["c", "c", "c", "c", "c", "c"])
        table_eyes_info.set_cols_dtype(['t', 't', 't', 't', 't', 't'])
        table_eyes_info.add_rows([["", "度数", "散光", "轴位", "下加光", "瞳距"],
                                  ["右眼：", self.SPH_RightEye, self.CYL_RightEye, self.AXIS_RightEye, self.DOWN_RightEye,
                                   self.DIST_RightEye],
                                  ["左眼：", self.SPH_LeftEye, self.CYL_LeftEye, self.AXIS_LeftEye, self.DOWN_LeftEye,
                                   self.DIST_LeftEye]])

        html += table_eyes_info.draw()
        html += "\n"
        html += split_line + "\n"

        # 条件打印：print=2， 需要打印产品信息
        if self.printType == "2":
            html += "镜片品牌：" + self.GlassName + "\n"
            html += "镜片品类：" + self.GlassType + "\n"
            html += "镜架品牌：" + self.GlassFrameName + "\n"
            html += "镜架型号：" + self.GlassFrameMode + "\n"

        # 添加手写8行空格
        html += "\n\n\n\n\n\n\n\n"
        
        html += "备注信息：" + self.Comments + "\n"
        html += "打印时间：" + self.printDateTime + "\n"
        html += split_line + "\n"
        html += "品牌故事：人类眼睛有着精妙的结构，以及丰富巨量的5.76亿级像素。《5.76眼镜臻选》来源于此，我们最初的设想是通过精准的验光，臻选合适的镜架以及切合个人需求的镜片，能让部分人重回5.76亿的像素。\n"
        html += split_line + "\n"
        html += "欢迎光临,谢谢惠顾!\n"
        html += "联系方式:13621603550\n"
        html += "门店地址:上海市浦东新区杨高中路2108号天物空间A幢2层A230室\n"     

        # 格式化HTML
        html = html.replace("=", "-")
        html = html.replace("\n", "<BR>")

        self.logcat(html)
        return html

    def logcat(self, log):
        """
        将需要打印内容记录日志
        :param log:打印内容
        :return:
        """
        # 文件路径
        filePath = self.filePath + "\\log.txt"
        log = time.strftime("%Y-%m-%d %H:%M:%S") + ":  \n-------------------------\n" + log
        with open(filePath, "a") as file:
            file.write(log + "\n\n")

    def signature(self, STIME):
        """
        HTML 接口请求签名
        :return:
        """
        s1 = sha1()
        s1.update((USER + UKEY + STIME).encode())
        return s1.hexdigest()

    def requestPrintApi(self, content):
        """
        云端打印接口请求
        :param content:需要打印内容的HTML格式
        :return:
        """
        STIME = str(int(time.time()))  # 不需要修改
        params = {
            'user': USER,
            'sig': self.signature(STIME),
            'stime': STIME,
            'apiname': 'Open_printMsg',  # 固定值,不需要修改
            'sn': SN,
            'content': content,
            'times': '1'  # 打印联数
        }
        response = requests.post(URL, data=params, timeout=30)
        code = response.status_code
        if code == 200:
            self.logcat(response.content)
            print(response.content)
        else:
            self.logcat("print error:" + response.content)
            print("print error:" + response.content)


def printData():
    # data = "马婷,13636347810,婆婆,远用,-9.5,,0,,0,,0,,33,,视可悦,1.67,邦尼,5458,物业企划,D:\print,1"
    if len(sys.argv) == 2:
        data = sys.argv[1]

        # 初始化用户数据
        userVisionData = UserVisionData(data)
        # userVisionData.printTextTable()

        # 格式化云端打印数据
        html_content = userVisionData.getHtmlData()
        # print(html_content)

        # 需要云端打印可以开放注释
        userVisionData.requestPrintApi(html_content)

    else:
        print("input parameters error...")


if __name__ == '__main__':
    printData()
