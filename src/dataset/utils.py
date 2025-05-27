import os
import cv2
import random
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from pathlib import Path
from skvideo.io import ffprobe, FFmpegReader, FFmpegWriter


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)


def non_max_suppression_face(prediction, conf_thres=0.5, iou_thres=0.45, classes=None, agnostic=False, labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 15  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    output = [torch.zeros((0, 16), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 15), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 15] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 15:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, landmarks, cls)
        if multi_label:
            i, j = (x[:, 15:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 15, None], x[i, 5:15] ,j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 15:].max(1, keepdim=True)
            x = torch.cat((box, conf, x[:, 5:15], j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 15:16] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]

    return output


class YoloFace():
    def __init__(self, pt_path='checkpoints/yolov5m-face.pt', confThreshold=0.5, nmsThreshold=0.45, device='cuda'):
        assert os.path.exists(pt_path)

        self.inpSize = 416
        self.conf_thres = confThreshold
        self.iou_thres = nmsThreshold
        self.test_device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(pt_path).to(self.test_device)
        self.last_w = 416
        self.last_h = 416
        self.grids = None

    @torch.no_grad()
    def detect(self, srcimg):

        h0, w0 = srcimg.shape[:2]
        r = self.inpSize / min(h0, w0)
        h1 = int(h0*r+31)//32*32
        w1 = int(w0*r+31)//32*32

        img = cv2.resize(srcimg, (w1,h1), interpolation=cv2.INTER_LINEAR)

        # Convert
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR to RGB

        # Run inference
        img = torch.from_numpy(img).to(self.test_device).permute(2,0,1)
        img = img.float()/255  # uint8 to fp16/32  0-1
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        if h1 != self.last_h or w1 != self.last_w or self.grids is None:
            grids = []
            for scale in [8,16,32]:
                ny = h1//scale
                nx = w1//scale
                yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
                grid = torch.stack((xv, yv), 2).view((1,1,ny, nx, 2)).float()
                grids.append(grid.to(self.test_device))
            self.grids = grids
            self.last_w = w1
            self.last_h = h1

        pred = self.model(img, self.grids).cpu()

        # Apply NMS
        det = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)[0]
        bboxes = np.zeros((det.shape[0], 4))
        kpss = np.zeros((det.shape[0], 5, 2))
        scores = np.zeros((det.shape[0]))
        det = det.cpu().numpy()

        for j in range(det.shape[0]):
            bboxes[j, 0] = det[j, 0] * w0/w1
            bboxes[j, 1] = det[j, 1] * h0/h1
            bboxes[j, 2] = det[j, 2] * w0/w1 - bboxes[j, 0]
            bboxes[j, 3] = det[j, 3] * h0/h1 - bboxes[j, 1]
            scores[j] = det[j, 4]
            kpss[j, :, :] = det[j, 5:15].reshape(5, 2) * np.array([[w0/w1,h0/h1]])
        
        return kpss, scores, bboxes


class VideoUtils(object):
    def __init__(self, video_path=None, output_video_path=None, bit_rate='origin', fps=25):
        if video_path is not None:
            meta_data = ffprobe(video_path)
            codec_name = 'libx264'
            color_space = meta_data['video'].get('@color_space')
            color_transfer = meta_data['video'].get('@color_transfer')
            color_primaries = meta_data['video'].get('@color_primaries')
            color_range = meta_data['video'].get('@color_range')
            pix_fmt = meta_data['video'].get('@pix_fmt')
            if bit_rate=='origin':
                bit_rate = meta_data['video'].get('@bit_rate')
            else:
                bit_rate=None
            if pix_fmt is None:
                pix_fmt = 'yuv420p'

            reader_output_dict = {'-r': str(fps)}
            writer_input_dict = {'-r': str(fps)}
            writer_output_dict = {'-pix_fmt': pix_fmt, '-r': str(fps), '-vcodec':str(codec_name)}
            writer_output_dict['-crf'] = '17'

            # if video has alpha channel, convert to bgra, uint16 to process
            if pix_fmt.startswith('yuva'):
                writer_input_dict['-pix_fmt'] = 'bgra64le'
                reader_output_dict['-pix_fmt'] = 'bgra64le'
            elif pix_fmt.endswith('le'):
                writer_input_dict['-pix_fmt'] = 'bgr48le'
                reader_output_dict['-pix_fmt'] = 'bgr48le'
            else:
                writer_input_dict['-pix_fmt'] = 'bgr24'
                reader_output_dict['-pix_fmt'] = 'bgr24'

            if color_range is not None:
                writer_output_dict['-color_range'] = color_range
                writer_input_dict['-color_range'] = color_range
            if color_space is not None:
                writer_output_dict['-colorspace'] = color_space
                writer_input_dict['-colorspace'] = color_space
            if color_primaries is not None:
                writer_output_dict['-color_primaries'] = color_primaries
                writer_input_dict['-color_primaries'] = color_primaries
            if color_transfer is not None:
                writer_output_dict['-color_trc'] = color_transfer
                writer_input_dict['-color_trc'] = color_transfer

            writer_output_dict['-sws_flags'] = 'full_chroma_int+bitexact+accurate_rnd'
            reader_output_dict['-sws_flags'] = 'full_chroma_int+bitexact+accurate_rnd'

            print(writer_input_dict)
            print(writer_output_dict)

            self.reader = FFmpegReader(video_path, outputdict=reader_output_dict)
        else:
            codec_name = 'libx264'
            bit_rate=None
            pix_fmt = 'yuv420p'

            reader_output_dict = {'-r': str(fps)}
            writer_input_dict = {'-r': str(fps)}
            writer_output_dict = {'-pix_fmt': pix_fmt, '-r': str(fps), '-vcodec':str(codec_name)}
            writer_output_dict['-crf'] = '17'

            # if video has alpha channel, convert to bgra, uint16 to process
            if pix_fmt.startswith('yuva'):
                writer_input_dict['-pix_fmt'] = 'bgra64le'
                reader_output_dict['-pix_fmt'] = 'bgra64le'
            elif pix_fmt.endswith('le'):
                writer_input_dict['-pix_fmt'] = 'bgr48le'
                reader_output_dict['-pix_fmt'] = 'bgr48le'
            else:
                writer_input_dict['-pix_fmt'] = 'bgr24'
                reader_output_dict['-pix_fmt'] = 'bgr24'

            writer_output_dict['-sws_flags'] = 'full_chroma_int+bitexact+accurate_rnd'
            print(writer_input_dict)
            print(writer_output_dict)

        if output_video_path is not None:
            self.writer = FFmpegWriter(output_video_path, inputdict=writer_input_dict, outputdict=writer_output_dict, verbosity=1)

    def getframes(self):
        return self.reader.nextFrame()

    def writeframe(self, frame):
        if frame is None:
            self.writer.close()
        else:
            self.writer.writeFrame(frame)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)

def save_videos_from_pil(pil_images, path, fps=8):
    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if save_fmt == ".mp4":
        video_cap = VideoUtils(output_video_path=path, fps=fps)
        for pil_image in pil_images:
            image_cv2 = np.array(pil_image)[:,:,[2,1,0]]
            video_cap.writeframe(image_cv2)
        video_cap.writeframe(None)

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
            optimize=False,
            lossless=True
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []

    for i, x in enumerate(videos):
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c h w)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_videos_from_pil(outputs, path, fps)


def get_head_exp_motion_bucketid(lmks, nose_index=2, max_value=127):
    exp_lmks = np.array([lmk - lmk[nose_index] for lmk in lmks])
    init_lmk = exp_lmks[0]
    scale = np.sqrt(((init_lmk.max(0) - init_lmk.min(0))**2).sum())
    exp_var = np.sqrt(((exp_lmks - exp_lmks.mean(0))**2).sum(2))
    exp_var = exp_var.mean()
    exp_var = exp_var/scale * 1024

    exp_var = int(exp_var)
    exp_var = max(exp_var, 0)
    exp_var = min(exp_var, max_value)

    head_poses = np.array([lmk[nose_index] for lmk in lmks])
    head_var = np.sqrt(((head_poses - head_poses.mean(0))**2).sum(1))
    head_var = head_var.mean()/scale  * 256

    head_var = int(head_var)
    head_var = max(head_var, 0)
    head_var = min(head_var, max_value)

    return head_var, exp_var