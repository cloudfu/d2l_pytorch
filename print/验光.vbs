Private Sub Worksheet_SelectionChange(ByVal Target As Range)
    
    '检测用户是否点击打印框
    
'    Dim row As Integer
'    Dim column As Integer
'    MsgBox (Cells(Target.row, Target.column).MergeCells)

   
   If Cells(Target.row, Target.Column).MergeCells = True Then
'        打印数据准备
        If Cells(Target.row, Target.Column).MergeArea.Cells(1, 1).Value = "打印1" Then
            Call sendPrintData(Target, "1")
        ElseIf Cells(Target.row, Target.Column).MergeArea.Cells(1, 1).Value = "打印2" Then
            Call sendPrintData(Target, "2")
        End If
   End If
    
End Sub

Private Sub sendPrintData(ByVal Target As Range, ByVal PrintType As String)

    '当前按钮对应行
    Dim row As Integer
    row = Target.row

    '============= 用户信息 ===========
    
    '用户姓名
    Dim idxUserName As String
    Dim UserName As String
    idxUserName = "B" & row
    UserName = Range(idxUserName).Value
'    MsgBox (UserName)

    '手机号
    Dim idxPhoneNum As String
    Dim PhoneNum As String
    idxPhoneNum = "C" & row
'    MsgBox (idxPhoneNum)
    PhoneNum = Range(idxPhoneNum).Value
'    MsgBox (PhoneNum)


    '验光单所属人
    Dim idxBillOwner As String
    Dim BillOwner As String
    idxBillOwner = "D" & row
'    MsgBox (idxBillOwner)
    BillOwner = Range(idxBillOwner).Value
    
    '配镜用途
    Dim idxPurpose As String
    Dim Purpose As String
    idxPurpose = "E" & row
'    MsgBox (idxBillOwner)
    Purpose = Range(idxPurpose).Value
    
    '============= 验光信息 ===========
    'SPH_右眼
    Dim idx_SPH_RightEye As String
    Dim SPH_RightEye As String
    idx_SPH_RightEye = "G" & row
    SPH_RightEye = Range(idx_SPH_RightEye).Value
    
    'SPH_左眼
    Dim idx_SPH_LeftEye As String
    Dim SPH_LeftEye As String
    idx_SPH_LeftEye = "G" & (row + 1)
    SPH_LeftEye = Range(idx_SPH_LeftEye).Value

    'CYL_右眼
    Dim idx_CYL_RightEye As String
    Dim CYL_RightEye As String
    idx_CYL_RightEye = "H" & row
    CYL_RightEye = Range(idx_CYL_RightEye).Value

    'CYL_左眼
    Dim idx_CYL_LeftEye As String
    Dim CYL_LeftEye As String
    idx_CYL_LeftEye = "H" & (row + 1)
    CYL_LeftEye = Range(idx_CYL_LeftEye).Value

    'AXIS_右眼
    Dim idx_AXIS_RightEye As String
    Dim AXIS_RightEye As String
    idx_AXIS_RightEye = "I" & row
    AXIS_RightEye = Range(idx_AXIS_RightEye).Value

    'AXIS_左眼
    Dim idx_AXIS_LeftEye As String
    Dim AXIS_LeftEye As String
    idx_AXIS_LeftEye = "I" & (row + 1)
    AXIS_LeftEye = Range(idx_AXIS_LeftEye).Value

    '下加光_右眼
    Dim idx_DOWN_RightEye As String
    Dim DOWN_RightEye As String
    idx_DOWN_RightEye = "J" & row
    DOWN_RightEye = Range(idx_DOWN_RightEye).Value

    '下加光_左眼
    Dim idx_DOWN_LeftEye As String
    Dim DOWN_LeftEye As String
    idx_DOWN_LeftEye = "J" & (row + 1)
    DOWN_LeftEye = Range(idx_DOWN_LeftEye).Value

    '瞳距_右眼
    Dim idx_DIST_RightEye As String
    Dim DIST_RightEye As String
    idx_DIST_RightEye = "K" & row
    DIST_RightEye = Range(idx_DIST_RightEye).Value

    '瞳距_左眼
    Dim idx_DIST_LeftEye As String
    Dim DIST_LeftEye As String
    idx_DIST_LeftEye = "K" & (row + 1)
    DIST_LeftEye = Range(idx_DIST_LeftEye).Value

    '验光日期
    Dim idx_OptometryData As String
    Dim OptometryData As String
    idx_OptometryData = "L" & (row)
    OptometryData = Range(idx_OptometryData).Value
    
    '============= 商品信息 ===========

    '镜片
    Dim idx_GlassName As String
    Dim GlassName As String
    idx_GlassName = "N" & row
    GlassName = Range(idx_GlassName).Value

    '镜片品类
    Dim idx_GlassType As String
    Dim GlassType As String
    idx_GlassType = "O" & row
    GlassType = Range(idx_GlassType).Value

    '镜架
    Dim idx_GlassFrameName As String
    Dim GlassFrameName As String
    idx_GlassFrameName = "P" & row
    GlassFrameName = Range(idx_GlassFrameName).Value

    '镜架型号
    Dim idx_GlassFrameModel As String
    Dim GlassFrameMode As String
    idx_GlassFrameModel = "Q" & row
    GlassFrameMode = Range(idx_GlassFrameModel).Value

    '备注
    Dim idx_Comments As String
    Dim Comments As String
    idx_Comments = "W" & row
    Comments = Range(idx_Comments).Value

    '============= 数据整合 & 发送打印数据 ===========
    
    Dim PrintData As String
    Dim char_splite As String

    char_splite = ","
    
    '用户信息
    PrintData = UserName & char_splite
    PrintData = PrintData & PhoneNum & char_splite
    PrintData = PrintData & BillOwner & char_splite
    PrintData = PrintData & Purpose & char_splite

    '验光信息
    PrintData = PrintData & SPH_RightEye & char_splite
    PrintData = PrintData & SPH_LeftEye & char_splite
    PrintData = PrintData & CYL_RightEye & char_splite
    PrintData = PrintData & CYL_LeftEye & char_splite
    PrintData = PrintData & AXIS_RightEye & char_splite
    PrintData = PrintData & AXIS_LeftEye & char_splite
    PrintData = PrintData & DOWN_RightEye & char_splite
    PrintData = PrintData & DOWN_LeftEye & char_splite
    PrintData = PrintData & DIST_RightEye & char_splite
    PrintData = PrintData & DIST_LeftEye & char_splite
    
    '商品信息
    PrintData = PrintData & GlassName & char_splite
    PrintData = PrintData & GlassType & char_splite
    PrintData = PrintData & GlassFrameName & char_splite
    PrintData = PrintData & GlassFrameMode & char_splite
    PrintData = PrintData & Comments & char_splite
    PrintData = PrintData & ThisWorkbook.Path & char_splite
    PrintData = PrintData & PrintType
    
    Dim shellCmd As String
    shellCmd = " python " & ThisWorkbook.Path + "\pyPrint.py " & PrintData
'    Range("B70").Value = shellCmd
        
'    WshShell.Run "c:\windows\system32\cmd.exe /K python d:\\pyPrint.py " & PrintData
    
    Set WshShell = CreateObject("Wscript.Shell")
    Shell "cmd /c" & shellCmd
     
End Sub


