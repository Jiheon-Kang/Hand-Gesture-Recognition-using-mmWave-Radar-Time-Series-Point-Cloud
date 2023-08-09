import struct
import glob
import os
from math import sin, cos

# =========================================================
# raw 데이터 위치
parent_dir = './NewData/AllData/1.raw/new/'
# 파싱한 txt 파일 저장 위치
extract_path = './NewData/AllData/2.parsed/new/'

# =========================================================

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
    header = {}
    header['seq'] = frame['header']['frameNumber']

    for tlv in frame['tlvs']:
        if tlv['type'] != 1020:
            continue

        unit = tlv['value'][0]
        pointCloud = tlv['value'][1]

        for i, p in enumerate(pointCloud):
            record = {}
            record['header'] = header
            record['point_id'] = i
            r = p['range'] * unit['rangeUnit']
            el = p['elevation'] * unit['elevationUnit']
            az = p['azimuth'] * unit['azimuthUnit']
            record['x'] = r * cos(el) * sin(az)
            record['y'] = r * cos(el) * cos(az)
            record['z'] = r * sin(el)
            record['range'] = r
            pointList.append(record)
        break

    if len(pointList) == 0:
        return

    return pointList

import json
# raw 파일을 파싱하여 얻은 정보를 out으로 저장하는 함수
def RawToParsedTxt(raw, out):
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

    frames = []
    min = 1000000
    max = 0
    total = 0
    except_cnt = 0
    for p in packets:
        # print('----------------------------')
        frame = getFrameData(p)
        if not frame:
            continue
        #print(json.dumps(frame, indent=3))
        #print(frame)

        try:

            numOfPoint = len(frame["tlvs"][0]["value"][1])
            # packetLenth = frame["header"]['totalPacketLen']
            total += numOfPoint
            if (max < numOfPoint): max = numOfPoint
            if (min > numOfPoint): min = numOfPoint

        except:

            except_cnt += 1
            # print(json.dumps(frame, indent=3))

        frames.append(frame)
    #print("==============================================")
    #print("데이터 손실율: ", (len(packets)-len(frames))/len(frames))
    #print("패킷 길이 통계 - 최소:", min, "최대:", max, "평균:", total/(len(frames)-except_cnt))

    print("Read \"" + raw + "\" and write parsed txt file ...")
    with open(out, 'w') as f:
        for frame in frames:
            pointData = getDataforHAR(frame)
            if not pointData:
                continue
            for d in pointData:
                for key, v in d.items():
                    str = key+':'
                    print(str, v, file=f)
                print('------------------------------', file=f)


files = sorted(glob.glob(os.path.join(parent_dir, '*.txt')))

for fn in files:
    out = os.path.basename(fn)
    out = os.path.join(extract_path, out)
    RawToParsedTxt(fn, out)
