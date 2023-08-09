import struct
import time
from math import sin, cos, sqrt
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import glob
import os
import random

np.random.seed(0)
random.seed(0)

def classify_spread_feature(data):
    # Spread
    before_frames_max_diff = data[data[:, 3] < 0.5][:, :3].max() - data[data[:, 3] < 0.5][:, :3].min()
    after_frame_max_diff = data[data[:, 3] > 0.5][:, :3].max() - data[data[:, 3] > 0.5][:, :3].min()

    before_frames_max_diff2 = data[data[:, 3] < 0.75][:, :3].max() - data[data[:, 3] < 0.75][:, :3].min()
    after_frame_max_diff2 = data[data[:, 3] > 0.75][:, :3].max() - data[data[:, 3] > 0.75][:, :3].min()

    before_frames_max_diff3 = data[data[:, 3] < 0.25][:, :3].max() - data[data[:, 3] < 0.25][:, :3].min()
    after_frame_max_diff3 = data[data[:, 3] > 0.25][:, :3].max() - data[data[:, 3] > 0.25][:, :3].min()

    if after_frame_max_diff - before_frames_max_diff > 0.05 and after_frame_max_diff2 - before_frames_max_diff2 > 0.05 \
            and after_frame_max_diff3 - before_frames_max_diff3 > 0.05:
        return 'Spread'
    elif after_frame_max_diff - before_frames_max_diff < -0.05 and after_frame_max_diff2 - before_frames_max_diff2 < -0.05 \
            and after_frame_max_diff3 - before_frames_max_diff3 < -0.05:
        return 'Spread'
    else:
        # 제스처 특징이 뚜렷하지 않으면 pass
        return

def classify_leftright_feature(data):
    # RIGHT LEFT
    before_frames_x_avg, after_frame_x_avg = data[data[:, 3] < 0.5].mean(axis=0)[0], \
                                             data[data[:, 3] > 0.5].mean(axis=0)[0]
    before_frames_x_avg2, after_frame_x_avg2 = data[data[:, 3] < 0.75].mean(axis=0)[0], \
                                             data[data[:, 3] > 0.75].mean(axis=0)[0]
    before_frames_x_avg3, after_frame_x_avg3 = data[data[:, 3] < 0.25].mean(axis=0)[0], \
                                               data[data[:, 3] > 0.25].mean(axis=0)[0]

    if after_frame_x_avg - before_frames_x_avg > 0.2 and after_frame_x_avg2 - before_frames_x_avg2 > 0.2 \
            and after_frame_x_avg3 - before_frames_x_avg3 > 0.2:
        return 'rightToLeft'
    elif after_frame_x_avg - before_frames_x_avg < -0.2 and after_frame_x_avg2 - before_frames_x_avg2 < -0.2 \
            and after_frame_x_avg3 - before_frames_x_avg3 < -0.2:
        return 'leftToRight'
    else:
        # 제스처 특징이 뚜렷하지 않으면 pass
        return

def classify_updown_feature(data):
    # RIGHT LEFT
    before_frames_z_avg, after_frame_z_avg = data[data[:, 3] < 0.5].mean(axis=0)[2], \
                                             data[data[:, 3] > 0.5].mean(axis=0)[2]

    before_frames_z_avg2, after_frame_z_avg2 = data[data[:, 3] < 0.75].mean(axis=0)[2], \
                                             data[data[:, 3] > 0.75].mean(axis=0)[2]

    before_frames_z_avg3, after_frame_z_avg3 = data[data[:, 3] < 0.25].mean(axis=0)[2], \
                                               data[data[:, 3] > 0.25].mean(axis=0)[2]

    if after_frame_z_avg - before_frames_z_avg > 0.2 and after_frame_z_avg2 - before_frames_z_avg2 > 0.2 \
            and after_frame_z_avg3 - before_frames_z_avg3 > 0.2:
        return 'downToUp'
    elif after_frame_z_avg - before_frames_z_avg < -0.2 and after_frame_z_avg2 - before_frames_z_avg2 < -0.2 \
            and after_frame_z_avg3 - before_frames_z_avg3 < -0.2:
        return 'upToDown'
    else:
        # 제스처 특징이 뚜렷하지 않으면 pass
        return

# =========================================================

seq_classify_functions = {
    'downToUp': None,
    'leftToRight': None,
    'rightToLeft': None,
    'Spread': None,
    'upToDown': None
}

# =========================================================

# raw 데이터 위치
parent_dir = '../../NewData/AllData/1.raw/old/'
# npz 파일 저장 위치
extract_path = '../../NewData/AllData/5.pointcloud/exp'

# 클래스
classes = ['diag-LeftToRight', 'diag-RightToLeft']

# 훈련/테스트셋 분할 비율
splitSet = True    # 데이터셋 분할 여부
test_size = 0.3     # 30%를 테스트셋으로 사용

# 포인트 클라우드 추출 범위
xLimit = 0.2    # -0.2m ~ 0.2m(좌우)
yLimit = 0.4    #    0m ~ 0.4m(정면거리)
zLimit = 0.2    # -0.2m ~ 0.2m(상하)

ROI_filter = True
coordinate_normalize = True
addSeqCh = True
coordinate_aug_range = 0.01
samplingNum = 64

save_visualize_data = False
visualize_aug_for_check = False

# =========================================================

# 시퀀스 처리
frames_together = 27 # 시퀀스 길이
sliding = 10 # 슬라이딩 윈도우

# =========================================================
"""
if ROI_filter:
    extract_path += "_ROI"
    if coordinate_normalize:
        extract_path += "norm"
"""
if addSeqCh:
    extract_path += "_4ch_aug"+str(coordinate_aug_range)+"_sample"+str(samplingNum)
else:
    extract_path += "_3ch_aug"+str(coordinate_aug_range)+"_sample"+str(samplingNum)

if sliding != 10:
    extract_path += "_tw"+str(sliding)
extract_path += "_p0.3"

if extract_path[-1] != "/":
    extract_path += "/"

if not os.path.exists(extract_path):
    os.makedirs(extract_path)
if save_visualize_data:
    for cls in classes:
        cls_dir = extract_path + cls
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)


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

def getFrameData_old(arr):
    if len(arr) < 48:
        return
    frameHeader = getFrameHeader_old(arr[:48])
    frame = {}
    frame['header'] = frameHeader
    packetLen = frameHeader['totalPacketLen']
    numTLVs = frameHeader['numTLVs']
    # print("packetLen: ", packetLen)
    if len(arr) < packetLen:
        # print('데이터 손실 발생')
        # print('받은 데이터 길이:', len(arr), '받아야 할 데이터 길이:', packetLen)
        return

    # TLVs 읽기
    if frameHeader['sync'] == hex(0x708050603040102):
        arr = arr[48:packetLen]
        tlvs = []
        tlvsLen = 0
        while (packetLen - 48 > tlvsLen):
            tlv = getTLV_old(arr[tlvsLen:])
            if not tlv:
                break
            tlvsLen += tlv['length']
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

def getFrameHeader_old(arr):
    header = {}
    headerMsg = struct.unpack('Q9IHH', bytearray(arr))
    header['sync'] = hex(headerMsg[0])
    header['version'] = headerMsg[1]
    header['totalPacketLen'] = headerMsg[2]
    header['platform'] = headerMsg[3]
    header['frameNumber'] = headerMsg[4]
    header['subFrameNumber'] = headerMsg[5]
    header['chirpProcMargin'] = headerMsg[6]
    header['frameProcMargin'] = headerMsg[7]
    header['trackProcTime'] = headerMsg[8]
    header['uartSentTime'] = headerMsg[9]
    header['numTLVs'] = headerMsg[10]
    header['checksum'] = headerMsg[11]
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

def getTLV_old(arr):
    # TLV Type: 6 = Point cloud
    tlv = {}
    TLVheader = struct.unpack('2I', bytearray(arr[0:8]))
    tlv['type'] = TLVheader[0]
    tlv['length'] = TLVheader[1]
    length = tlv['length'] # tlv 헤더의 length는 헤더 길이(8바이트)를 제외한 value 길이만을 의미
    type = tlv['type']
    if type == 6:
        tlvHeaderLen = 8
        pointUnitLen = 20
        pointStructLen = 8

        numOfPoints = int((length - tlvHeaderLen - pointUnitLen) / pointStructLen)
        # print("numOfPoints", numOfPoints)
        points = []
        pointUnit = getPointUnit_old(arr[tlvHeaderLen : tlvHeaderLen+pointUnitLen])
        arr = arr[tlvHeaderLen+pointUnitLen:]
        for i in range(numOfPoints):
            point = getPointCloud_old(arr[i*pointStructLen:(i+1)*pointStructLen])
            points.append(point)
        tlv['value'] = [pointUnit, points]
        return tlv
    else:
        # print('TLVtypeError', type)
        return

def getPointUnit(arr):
    unit = {}
    pointUnit = struct.unpack('3f', bytearray(arr))
    unit['elevationUnit'] = pointUnit[0]
    unit['azimuthUnit'] = pointUnit[1]
    unit['rangeUnit'] = pointUnit[2]
    return unit

def getPointUnit_old(arr):
    unit = {}
    pointUnit = struct.unpack('5f', bytearray(arr))
    unit['elevationUnit'] = pointUnit[0]
    unit['azimuthUnit'] = pointUnit[1]
    unit['dopplerUnit'] = pointUnit[2]
    unit['rangeUnit'] = pointUnit[3]
    unit['snrUnit'] = pointUnit[4]
    return unit

def getPointCloud(arr):
    point = {}
    pointStruct = struct.unpack('bbH', bytearray(arr))
    point['elevation'] = pointStruct[0]
    point['azimuth'] = pointStruct[1]
    point['range'] = pointStruct[2]
    return point

def getPointCloud_old(arr):
    point = {}
    pointStruct = struct.unpack('bbhHH', bytearray(arr))
    point['elevation'] = pointStruct[0]
    point['azimuth'] = pointStruct[1]
    point['doppler'] = pointStruct[2]
    point['range'] = pointStruct[3]
    point['snr'] = pointStruct[4]
    return point

def getDataforHAR(frame):
    if 'tlvs' not in frame:
        return

    pointList = []

    for tlv in frame['tlvs']:
        if tlv['type'] != 6:
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
            if ROI_filter:
                if abs(x) > xLimit or abs(y) > yLimit or abs(z) > zLimit:
                    continue
                if coordinate_normalize:
                    x = (x + xLimit) / (2 * xLimit)
                    y /= yLimit
                    z = (z + zLimit) / (2 * zLimit)
            pointList.append([x, y, z])
        break

    return pointList

def RawToPoints(raw):

    # 기록한 txt 파일을 패킷 단위로 끊어 읽기 ------------------------------------------------------------------
    packets = []
    with open(raw, 'r') as f:
        str = f.read()
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
        frame = getFrameData_old(p)
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
    print("Convert bytes to point data structure")
    # print("프레임당 포인트 갯수 통계")
    # print(f"MAX: {max} MIN: {min} AVG: {total / len(frames)}")
    print(f"complete - 손실 패킷 수: {except_cnt}")


    return pointframes


def PointsToSeqScenes(points, seqClass, point_sample_num=64):
    print("---------------------------------")
    print(len(points), " frames to sequence Scene Data")
    datas = []
    labels = []

    # 포인트 통계용
    MIN = 100000
    MAX = 0
    AVG = 0

    i = 0
    while i + frames_together <= len(points):
        # print(i + frames_together, len(points))
        local_data = []
        for j in range(frames_together):
            for point in points[i+j]:
                if(addSeqCh):
                    point4ch = point + [j/(frames_together-1)]
                    # print(point4ch)
                    local_data.append(point4ch)
                else:
                    local_data.append(point)
        i = i + sliding

        data = np.array(local_data)
        # 시퀀스 잘못 잘렸는지 검사 (앞뒤 혹은 전체가 텅빈 시퀀스인지)
        if addSeqCh:
            if data.shape[0] == 0 or data[data[:, 3] > 0.5].shape[0] == 0 or data[data[:, 3] < 0.5].shape[0] == 0:
                continue
        else:
            if data.shape[0] == 0:
                continue
        """
        if data.shape[0] == 0 or data[data[:, 3] > 0.5].shape[0] == 0 or data[data[:, 3] < 0.5].shape[0] == 0 \
                or data[data[:, 3] > 0.75].shape[0] == 0 or data[data[:, 3] < 0.75].shape[0] == 0 \
                or data[data[:, 3] > 0.25].shape[0] == 0 or data[data[:, 3] < 0.25].shape[0] == 0:
            # print("wrong sequence")
            continue
        """
        original_point_num = data.shape[0]

        # sampling (aug+)
        data = pointSampling(data, point_sample_num)

        """
        # seq feature filtering
        get_seq_feature_label = seq_classify_functions[seqClass]
        if get_seq_feature_label:
            label = get_seq_feature_label(data)

            # 제스처 특징이 뚜렷하지 않아 label return 값이 null이면 pass
            if not label:
                continue
        else:
            label = seqClass
        """
        label = seqClass

        # 통계 계산
        if original_point_num < MIN:
            MIN = original_point_num
        if original_point_num > MAX:
            MAX = original_point_num
        AVG += original_point_num

        # seq feature 추출 완료, 리스트에 추가
        labels.append(label)
        datas.append(data)

    del points, local_data, data

    # 통계 출력
    print(len(labels))
    print(seqClass + " PointCloud Lenth - MIN", MIN, "MAX", MAX, "AVG", AVG / len(labels))

    datas = np.array(datas)

    return datas, labels

def pointSampling(ndarr, sample_num):
    # print(ndarr.shape[0], sample_num)
    if visualize_aug_for_check:
        fig = plt.figure()
        ax = plt.subplot(111, projection='3d')
        if addSeqCh:
            ax.scatter(ndarr[:, 0], ndarr[:, 1], ndarr[:, 2], c=ndarr[:, 3])
        else:
            ax.scatter(ndarr[:, 0], ndarr[:, 1], ndarr[:, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(5, 80)
        ax.set_title("BEFORE augment sampling")
        plt.show()

    while (ndarr.shape[0] < sample_num):
        auged_ndarr = ndarr.copy()
        auged_ndarr[:, :3] += np.random.uniform(-coordinate_aug_range, coordinate_aug_range, size=(ndarr.shape[0], 3))
        ndarr = np.vstack([ndarr, auged_ndarr])

    pointNums = ndarr.shape[0]


    if addSeqCh:

        seq_dim_unique = np.unique(ndarr[:, 3])

        sampleNumPerFrame = sample_num // len(seq_dim_unique)
        sampleNumRemaineder = sample_num % len(seq_dim_unique)

        sampled_idx = []
        # 각 프레임별로 일정 개수 추출
        for seq_dim in seq_dim_unique:
            seq_row_idxs = np.where(ndarr[:, 3] == seq_dim)[0]
            if len(seq_row_idxs) > sampleNumPerFrame:
                sampled_idx += list(np.random.choice(seq_row_idxs, size=sampleNumPerFrame, replace = False))
            else:
                sampled_idx += list(seq_row_idxs)
                sampleNumRemaineder += sampleNumPerFrame - len(seq_row_idxs)
        # 나머지 개수는 프레임 상관없이 랜덤 추출
        # print(pointNums, len(seq_dim_unique), sampleNumPerFrame, sampleNumRemaineder)
        sampled_idx += random.sample(list(set(range(pointNums)) - set(sampled_idx)), sampleNumRemaineder)
        sampledPoints = ndarr[sampled_idx]

    else:
        sampledPoints = np.random.permutation(ndarr)[:sample_num]

    if sampledPoints.shape[0] != sample_num:
        print("Something Wrong")

    if coordinate_normalize:
        sampledPoints[sampledPoints > 1.0] = 1.0
        sampledPoints[sampledPoints < 0] = 0

    if visualize_aug_for_check:
        fig = plt.figure()
        ax = plt.subplot(111, projection='3d')
        if addSeqCh:
            ax.scatter(sampledPoints[:, 0], sampledPoints[:, 1], sampledPoints[:, 2], c=sampledPoints[:, 3])
        else:
            ax.scatter(sampledPoints[:, 0], sampledPoints[:, 1], sampledPoints[:, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(5, 80)
        ax.set_title("AFTER augment sampling")
        plt.show()

    # print(sampledPoints[sampledPoints[:, 3] < 0.5].shape, sampledPoints[sampledPoints[:, 3] > 0.5].shape)
    if addSeqCh:
        if ndarr[ndarr[:, 3] > 0.5].shape[0] == 0:
            points_per_frame = []
            # 각 프레임별로 일정 개수 추출
            for seq_dim in seq_dim_unique:
                points_per_frame.append(np.where(ndarr[:, 3] == seq_dim)[0].shape[0])
            print(seq_dim_unique)
            print("after frames invalid", points_per_frame)
            print("------------------")

    return sampledPoints

def parse_raw_files_to_seqScene(parent_dir, cls, file_ext='*.txt'):
    if addSeqCh:
        features =np.empty((0, samplingNum, 4))
    else:
        features = np.empty((0, samplingNum, 3))

    labels = []

    files = sorted(glob.glob(os.path.join(parent_dir, cls + file_ext)))
    for fn in files:
        print(fn)
        points = RawToPoints(fn)
        file_features, file_labels = PointsToSeqScenes(points, cls, samplingNum)

        if save_visualize_data:
            for idx in range(50):
                data = file_features[idx]
                real_cls = file_labels[idx]
                fig = plt.figure()
                ax = plt.subplot(111, projection='3d')
                if addSeqCh:
                    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=data[:, 3])
                else:
                    ax.scatter(data[:, 0], data[:, 1], data[:, 2])
                ax.set_xlim(0, 1.0)
                ax.set_ylim(0, 1.0)
                ax.set_zlim(0, 1.0)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                title = ax.set_title(cls + str(idx))
                ax.view_init(10, 80)
                if cls in ['downToUp', 'upToDown']:
                    ax.view_init(10, 10)
                elif cls in ['leftToRight', 'rightToLeft', 'Spread']:
                    ax.view_init(5, 80)

                file_name = extract_path+real_cls+'/'+real_cls + str(idx)

                plt.savefig(file_name+'.png')
                plt.close(fig)

        # print(features.shape, file_feature.shape)
        features = np.vstack([features, file_features])
        labels += file_labels

        del points, file_features, file_labels

    return features, labels


data_size_per_class = dict()
cls_features = []
for cls in classes:
    print("== " + cls + " =====================================")
    features, labels = parse_raw_files_to_seqScene(parent_dir, cls)

    print("---------------------------------")
    print("Total Features shape:", features.shape, "Labels shape:", len(labels))
    data_size_per_class[cls] = features.shape[0]
    # cls_features.append((features, labels))

    if not splitSet:
        Data_path = extract_path + cls
        np.savez(Data_path, features, labels)
        del features, labels
    else:
        Train_Data_path = extract_path + "Train_" + cls
        Test_Data_path = extract_path + "Test_" + cls
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=1)
        del features, labels

        print("Train Data Shape:", X_train.shape)
        print("Test Data Shape:", X_test.shape, "\n\n")
        if not save_visualize_data:
            np.savez(Train_Data_path, X_train, y_train)
            np.savez(Test_Data_path, X_test, y_test)

        del X_train, X_test, y_train, y_test

print("=======================================")
print(len(data_size_per_class), "Classes")

for cls in data_size_per_class:
    print(cls, data_size_per_class[cls])

"""
min_size = int(min(data_size_per_class.values()))
print("=======================================")


for idx, (features, labels) in enumerate(cls_features):
    cls = classes[idx]

    shuffled_idx = np.arange(features.shape[0])
    np.random.shuffle(shuffled_idx)

    features = features[shuffled_idx][:min_size]
    labels = [labels[x] for x in shuffled_idx][:min_size]

    Train_Data_path = extract_path + "Train_" + cls
    Test_Data_path = extract_path + "Test_" + cls
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=1)
    del features, labels

    print("Train Data Shape:", X_train.shape)
    print("Test Data Shape:", X_test.shape, "\n\n")
    if not save_visualize_data:
        np.savez(Train_Data_path, X_train, y_train)
        np.savez(Test_Data_path, X_test, y_test)

    del X_train, X_test, y_train, y_test

"""

