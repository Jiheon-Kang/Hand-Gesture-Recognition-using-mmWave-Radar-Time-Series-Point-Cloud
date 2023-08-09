import struct
import glob
import os

# raw 데이터 위치
parent_dir = './NewData/AllData/1.raw/old/'
# parsed txt 파일 저장 위치
extract_path = './NewData/AllData/2.parsed/old/'

frame = []

# 1 프레임에 대한 포인트 클라우드 데이터 전체를 파싱하여 반환하는 함수
def getFrameData(arr):
    if len(arr) < 48:
        return
    frameHeader = getFrameHeader(arr[:48])
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
            tlv = getTLV(arr[tlvsLen:])
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
    # TLV Type: 6 = Point cloud
    tlv = {}
    TLVheader = struct.unpack('2I', bytearray(arr[0:8]))
    tlv['type'] = TLVheader[0]
    tlv['length'] = TLVheader[1]
    length = tlv['length']  # tlv 헤더의 length는 헤더 길이(8바이트)를 제외한 value 길이만을 의미
    type = tlv['type']
    if type == 6:
        tlvHeaderLen = 8
        pointUnitLen = 20
        pointStructLen = 8

        numOfPoints = int((length - tlvHeaderLen - pointUnitLen) / pointStructLen)
        # print("numOfPoints", numOfPoints)
        points = []
        pointUnit = getPointUnit(arr[tlvHeaderLen: tlvHeaderLen + pointUnitLen])
        arr = arr[tlvHeaderLen + pointUnitLen:]
        for i in range(numOfPoints):
            point = getPointCloud(arr[i * pointStructLen:(i + 1) * pointStructLen])
            points.append(point)
        tlv['value'] = [pointUnit, points]
        return tlv
    else:
        # print('TLVtypeError', type)
        return

def getPointUnit(arr):
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
    pointStruct = struct.unpack('bbhHH', bytearray(arr))
    point['elevation'] = pointStruct[0]
    point['azimuth'] = pointStruct[1]
    point['doppler'] = pointStruct[2]
    point['range'] = pointStruct[3]
    point['snr'] = pointStruct[4]
    return point

def getTargetObject(arr):
    target = {}
    object = list(struct.unpack('I9f', bytearray(arr[:40])))
    ec = struct.unpack('16f', bytearray(arr[40:104]))
    objadd = struct.unpack('2f', bytearray(arr[104:]))
    object.append(ec)
    object += objadd
    target['tid'] = object[0]
    target['posX'] = object[1]
    target['posY'] = object[2]
    target['posZ'] = object[3]
    target['velX'] = object[4]
    target['velY'] = object[5]
    target['velZ'] = object[6]
    target['accX'] = object[7]
    target['accY'] = object[8]
    target['accZ'] = object[9]
    target['ec'] = object[10]
    target['g'] = object[11]
    target['confidenceLevel'] = object[12]
    return target


from math import sin, cos, sqrt
def getDataforGRF(frame):
    datas = []
    header = {}
    header['seq'] = frame['header']['frameNumber']
    if frame['header']['numTLVs'] < 2 or 'tlvs' not in frame:
        return
    for tlv in frame['tlvs']:
        if tlv['type']!=6:
            continue
        unit = tlv['value'][0]
        points = tlv['value'][1]
        for i, p in enumerate(points):
            record = {}
            record['header'] = header
            record['point_id'] = i
            r = p['range'] * unit['rangeUnit']
            el = p['elevation'] * unit['elevationUnit']
            az = p['azimuth'] * unit['azimuthUnit']
            record['azimuth'] = az
            record['elevation'] = el
            record['range'] = r
            record['doppler'] = p['doppler'] * unit['dopplerUnit']
            record['snr'] = p['snr'] * unit['snrUnit']
            datas.append(record)
    return datas

def getDataforHAR(frame):
    datas = []
    header = {}
    header['seq'] = frame['header']['frameNumber']
    if frame['header']['numTLVs'] < 2 or 'tlvs' not in frame:
        return
    for tlv in frame['tlvs']:
        if tlv['type']!=6:
            continue
        unit = tlv['value'][0]
        points = tlv['value'][1]
        for i, p in enumerate(points):
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
            record['doppler'] = p['doppler'] * unit['dopplerUnit']
            record['snr'] = p['snr'] * unit['snrUnit']
            datas.append(record)
    return datas



# raw 파일을 파싱하여 얻은 정보를 out으로 저장하는 함수
def RawToData(raw, out):

    packets = []
    print(raw)
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
    for p in packets:
        # print('----------------------------')
        frame = getFrameData(p)
        if not frame:
            continue
        # print(frame)
        frames.append(frame)

    print('lost-packet', len(packets)-len(frames))


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

def RawToGRFData(raw, out):
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
    for p in packets:
        print('----------------------------')
        frame = getFrameData(p)
        if not frame:
            continue
        # print(frame)
        frames.append(frame)

    print(len(frames), len(packets))


    with open(out, 'w') as f:
        for frame in frames:
            pointData = getDataforGRF(frame)
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
    RawToData(fn, out)
