import struct
import time
from math import sin, cos, sqrt
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import glob
import os

# =========================================================
# raw 데이터 위치
parent_dir = './NewData/AllData/1.raw/new/'
# npz 파일 저장 위치
extract_path = './NewData/AllData/4.proj/'

# 클래스
classes = ['Circle', 'Spread', 'Spin', 'ForwardBack', 'downToUp', 'upToDown', 'leftToRight', 'rightToLeft']

# 훈련/테스트셋 분할 비율
test_size = 0.3     # 30%를 테스트셋으로 사용

# 포인트 클라우드 추출 범위
xLimit = 0.2    # -0.2m ~ 0.2m(좌우)
yLimit = 0.4    #    0m ~ 0.4m(정면거리)
zLimit = 0.2    # -0.2m ~ 0.2m(상하)

# =========================================================

# 이미지 크기
width = 32
height = 32

# 이미지 시퀀스 처리
frames_together = 27 # 시퀀스 길이
sliding = 10 # 슬라이딩 윈도우

# =========================================================

"""
if sliding != 10:
    extract_path += "_tw"+str(sliding)

extract_path += "_p0.3"
"""

if extract_path[-1] != "/":
    extract_path += "/"


if not os.path.exists(extract_path):
    os.makedirs(extract_path)

frame = []

# 1 프레임에 대한 포인트 클라우드 데이터 전체를 파싱하여 반환하는 함수
def getFrameData(arr):
    if len(arr) < 40:
        print("frame header lost")
        return

    frame = {}

    # 프레임 헤더 읽기
    frameHeader = getFrameHeader(arr[:40])
    frame['header'] = frameHeader
    packetLen = frameHeader['totalPacketLen']
    numTLVs = frameHeader['numTLVs']

    if len(arr) != packetLen:
        # print("frame lost "+"( 받은 데이터 길이:", len(arr), "받아야 할 데이터 길이:", packetLen, ")")
        return

    # TLVs 읽기
    if frameHeader['sync'] == hex(0x708050603040102):
        arr = arr[40:]
        if numTLVs == 0:
            # print("No TLVs!")
            return frame # frame에 'tlvs'가 존재하지 않음 -> 감지한 포인트 클라우드 없음
        elif numTLVs > 1:
            print(f"there are {numTLVs} TLVs!")
        tlvs = []
        read_pos = 0
        for i in range(numTLVs):
            tlv = getTLV(arr[read_pos:])
            if not tlv:
                continue
            read_pos = tlv['length']+8
            tlvs.append(tlv)
        frame['tlvs'] = tlvs
        return frame
    else:
        print('sync error')
        return

# 1 프레임에 대한 프레임 헤더 정보를 반환하는 함수
def getFrameHeader(arr):
    header = {}
    headerMsg = struct.unpack('Q8I', bytearray(arr))
    header['sync'] = hex(headerMsg[0])
    header['version'] = headerMsg[1]
    header['totalPacketLen'] = headerMsg[2]
    header['platform'] = headerMsg[3]
    header['frameNumber'] = headerMsg[4]
    header['time'] = headerMsg[5]
    header['numDetectedObj'] = headerMsg[6]
    header['numTLVs'] = headerMsg[7]
    header['subFrameNumber'] = headerMsg[8]
    return header

# 1 프레임에 포착된 포인트 클라우드에 대한 정보를 반환하는 함수
def getTLV(arr):
    # TLV Type: 1020 = Point cloud
    tlv = {}
    TLVheader = struct.unpack('2I', bytearray(arr[0:8]))
    tlv['type'] = TLVheader[0]
    tlv['length'] = TLVheader[1]
    length = tlv['length'] # tlv 헤더의 length는 헤더 길이(8바이트)를 제외한 value 길이만을 의미
    type = tlv['type']
    if type == 1020:
        tlvHeaderLen = 8
        pointUnitLen = 12
        pointStructLen = 4

        numOfPoints = int((length - tlvHeaderLen - pointUnitLen) / pointStructLen)
        # print("numOfPoints", numOfPoints)
        points = []
        pointUnit = getPointUnit(arr[tlvHeaderLen : tlvHeaderLen+pointUnitLen])
        arr = arr[tlvHeaderLen+pointUnitLen:]
        for i in range(numOfPoints):
            point = getPointCloud(arr[i*pointStructLen:(i+1)*pointStructLen])
            points.append(point)
        tlv['value'] = [pointUnit, points]
        return tlv
    else:
        print('TLVtypeError', type)
        return

def getPointUnit(arr):
    unit = {}
    pointUnit = struct.unpack('3f', bytearray(arr))
    unit['elevationUnit'] = pointUnit[0]
    unit['azimuthUnit'] = pointUnit[1]
    unit['rangeUnit'] = pointUnit[2]
    return unit

def getPointCloud(arr):
    point = {}
    pointStruct = struct.unpack('bbH', bytearray(arr))
    point['elevation'] = pointStruct[0]
    point['azimuth'] = pointStruct[1]
    point['range'] = pointStruct[2]
    return point

def getDataforHAR(frame):
    if 'tlvs' not in frame:
        return

    pointList = []

    for tlv in frame['tlvs']:
        if tlv['type'] != 1020:
            continue

        unit = tlv['value'][0]
        pointCloud = tlv['value'][1]

        for p in pointCloud:
            r = p['range'] * unit['rangeUnit']              # range decode
            el = p['elevation'] * unit['elevationUnit']     # elevation decode
            az = p['azimuth'] * unit['azimuthUnit']         # azimuth decode
            x = r * cos(el) * sin(az)
            y = r * cos(el) * cos(az)
            z = r * sin(el)
            if abs(x) > xLimit or abs(y) > yLimit or abs(z) > zLimit:
                continue

            # normalize coordinate
            x = (x + xLimit) / (2 * xLimit)
            y /= yLimit
            z = (z + zLimit) / (2 * zLimit)

            pointList.append([x, y, z])
        break

    if len(pointList) == 0:
        return

    return pointList

def RawToPoints(raw):

    # 기록한 txt 파일을 패킷 단위로 끊어 읽기 ------------------------------------------------------------------
    packets = []
    with open(raw, 'r') as f:
        str = f.read()
        str = str.replace("\n", "")
        str = str.split('2 1 4 3 6 5 8 7 ')
        str = list(filter(lambda s: s != '', str))
        str = list(map(lambda s: '2 1 4 3 6 5 8 7 '+s, str))
        for s in str:
            s = s.split()
            packet = []
            for c in s:
                packet.append(int(c))
            packets.append(packet)

    # 포인트 갯수 통계 확인용으로 사용한 변수 - 변환에는 필요 X --------
    min = 1000000  # 최소 포인트 갯수
    max = 0  # 최대 포인트 갯수
    total = 0  # 평균 포인트 갯수 계산용
    except_cnt = 0  # 깨지는 프레임 패킷 갯수 세기용
    # ---------------------------------------------------------

    # 전송받은 바이트 parsing하여 프레임 헤더+포인트클라우드TLV정보 읽어오기 ----------------------------------------
    frames = []
    for p in packets:
        frame = getFrameData(p)
        if not frame:
            except_cnt += 1
            continue

        # print(json.dumps(frame, indent=3))
        # print(frame)

        frames.append(frame)

    # 프레임 정보로부터 관심 영역 내에 있는 포인트 클라우드의 x,y,z 좌표만 가져오기 ----------------------------------------
    pointframes = []
    for frame in frames:
        pointList = getDataforHAR(frame)
        if not pointList:
            pointframes.append([])
            numOfPoint = 0
        else:
            pointframes.append(pointList)
            numOfPoint = len(pointList)

        # 포인트 갯수 통계 계산 -----------------------------------
        total += numOfPoint
        if (max < numOfPoint): max = numOfPoint
        if (min > numOfPoint): min = numOfPoint
        # --------------------------------------------------------

    # 포인트 갯수 통계 -------------------------------------------------------------------------------------------
    print("---------------------------------")
    print("프레임당 포인트 갯수 통계")
    print(f"MAX: {max} MIN: {min} AVG: {total / len(frames)}")
    print(f"손실 패킷 수: {except_cnt}")
    print("---------------------------------")

    return pointframes

def PointsToProjection(points, wid, hei):
    pixels = []

    for f in points:
        pixel = np.zeros([wid, hei])

        if len(f) == 0:
            pixels.append(pixel)
            continue

        f = np.array(f)

        f[:, 0] *= (wid - 1)      # x coordinate
        f[:, 2] *= (hei - 1)      # z coordinate
        f[:, 1] = 1 - f[:, 1]                   # y coordinate
        for p in f:
            pixel[round(p[2])][round(p[0])] += p[1]
        # pixel = pixel ** 0.75
        pixel /= pixel.max()
        pixels.append(pixel)

        # plt.imshow(pixel, cmap='Greys')
        # plt.show()

    pixels = np.array(pixels)

    train_data = []

    i = 0
    while i + frames_together <= pixels.shape[0]:
        local_data = []
        for j in range(frames_together):
            local_data.append(pixels[i + j])

        train_data.append(local_data)
        i = i + sliding

    train_data = np.array(train_data)

    del points, pixels

    return train_data


def parse_raw_files(parent_dir, cls, file_ext='*.txt'):
    features =np.empty((0, frames_together, width, height))
    labels = []

    files = sorted(glob.glob(os.path.join(parent_dir, cls + file_ext)))
    for fn in files:
        print(fn)
        points = RawToPoints(fn)
        train_data = PointsToProjection(points, width, height)
        features=np.vstack([features,train_data])
        for i in range(train_data.shape[0]):
            labels.append(cls)

        del points, train_data

    return features, labels



for cls in classes:
    print(cls + " ==================================================")
    features, labels = parse_raw_files(parent_dir, cls)

    Train_Data_path = extract_path + "Train_" + cls
    Test_Data_path = extract_path + "Test_" + cls

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=1)
    del features, labels

    print("Train Data Shape:", X_train.shape)
    print("Test Data Shape:", X_test.shape)

    np.savez(Train_Data_path, X_train, y_train)
    np.savez(Test_Data_path, X_test, y_test)

    del X_train, X_test, y_train, y_test