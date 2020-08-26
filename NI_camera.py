import ctypes
import nivision
import numpy as np
import struct

from library import cam_x, cam_y, cam_offset_x, cam_offset_y

offsetX = cam_offset_x
offsetY = cam_offset_y
total_pixels = cam_x * cam_y
fmt = '>' + str(total_pixels) + 'B'

cam = ctypes.c_char_p(b"cam0")
id = nivision.IMAQdxOpenCamera(cam, nivision.IMAQdxCameraControlModeController)
img = nivision.imaqCreateImage(nivision.ImageType(0))
nivision.IMAQdxSetAttribute(id, nivision.IMAQdxAttributeOffsetX, offsetX)
nivision.IMAQdxSetAttribute(id, nivision.IMAQdxAttributeOffsetY, offsetY)
nivision.IMAQdxSetAttribute(id, nivision.IMAQdxAttributeWidth, cam_x)
nivision.IMAQdxSetAttribute(id, nivision.IMAQdxAttributeHeight, cam_y)
nivision.IMAQdxSetAttribute(id, b'CameraAttributes::Shutter::Mode', 0)
nivision.IMAQdxSetAttribute(id, b'CameraAttributes::Shutter::Value', 10)
nivision.IMAQdxSetAttribute(id, b'CameraAttributes::FrameRate::Mode', 0)
nivision.IMAQdxSetAttribute(id, b'CameraAttributes::FrameRate::Value', 200)
nivision.IMAQdxConfigureGrab(id)

def capture():
   nivision.IMAQdxGrab(id, img, 1)
   picture = nivision.imaqImageToArray(img)
   data = struct.unpack(fmt, picture[0])
   data = np.asarray(data)
   data = np.reshape(data, (cam_y, cam_x))
   return data
