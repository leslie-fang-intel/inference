import os
from argparse import ArgumentParser
from utils import DefaultBoxes, Encoder, COCODetection
from base_model import Loss
from utils import SSDTransformer
from ssd_r34 import SSD_R34
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import numpy as np

if os.environ.get('USE_IPEX') == "1":
    import intel_pytorch_extension as ipex


def parse_args():
    parser = ArgumentParser(description="Train Single Shot MultiBox Detector"
                                        " on COCO")
    parser.add_argument('--data', '-d', type=str, default='../coco',
                        help='path to test and training data files')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use available GPUs')
    parser.add_argument('--seed', '-s', type=int,
                        help='manually set random seed for torch')
    parser.add_argument('--device', '-did', type=int,
                        help='device id')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                        help='set batch size of valuation, default is 32')
    parser.add_argument('--iteration', '-iter', type=int, default=None,
                        help='set the iteration of inference, default is None')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--threshold', '-t', type=float, default=0.20,
                        help='stop training early at threshold')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='path to model checkpoint file, default is None')
    parser.add_argument('--image-size', default=[1200,1200], type=int, nargs='+',
                        help='input image sizes (e.g 1400 1400,1200 1200')
    parser.add_argument('--strides', default=[3,3,2,2,2,2], type=int, nargs='+',
                        help='stides for ssd model must include 6 numbers')
    parser.add_argument('--use-fp16', action='store_true')
    parser.add_argument('--ipex', action='store_true', default=False,
                        help='use intel pytorch extension')
    parser.add_argument('--int8', action='store_true', default=False,
                        help='enable ipex int8 path')
    parser.add_argument('--jit', action='store_true', default=False,
                        help='enable ipex jit path')
    parser.add_argument('--calibration', action='store_true', default=False,
                        help='doing int8 calibration step')
    parser.add_argument('--configure', default='configure.json', type=str, metavar='PATH',
                        help='path to int8 configures, default file name is configure.json')
    parser.add_argument("--dummy", action='store_true',
                        help="using  dummu data to test the performance of inference")
    parser.add_argument('-w', '--warmup-iterations', default=0, type=int, metavar='N',
                        help='number of warmup iterations to run')
    parser.add_argument('--autocast', action='store_true', default=False,
                        help='enable autocast')
    parser.add_argument('--profile', action='store_true', default=False,
                        help='enable profile')
    return parser.parse_args()


def show_memusage(device=0):
    import gpustat
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("{}/{}".format(item["memory.used"], item["memory.total"]))

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def dboxes_R34_coco(figsize,strides):
    ssd_r34=SSD_R34(81,strides=strides)
    synt_img=torch.rand([1,3]+figsize)
    _,_,feat_size =ssd_r34(synt_img, extract_shapes = True)
    steps=[(int(figsize[0]/fs[0]),int(figsize[1]/fs[1])) for fs in feat_size]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [(int(s*figsize[0]/300),int(s*figsize[1]/300)) for s in [21, 45, 99, 153, 207, 261, 315]]
    aspect_ratios =  [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes

def coco_eval(model, val_dataloader, cocoGt, encoder, inv_map, args):
    from pycocotools.cocoeval import COCOeval
    device = args.device
    threshold = args.threshold
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    model.eval()

    ret = []

    inference_time = AverageMeter('InferenceTime', ':6.3f')
    decoding_time = AverageMeter('DecodingTime', ':6.3f')

    progress = ProgressMeter(
        args.iteration if args.dummy else len(val_dataloader),
        [inference_time, decoding_time],
        prefix='Test: ')

    start = time.time()
    if args.int8:
        model = model.eval()
        # disable PyTorch JIT profiling
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        # disable IPEX JIT optimization
        ipex.core.disable_jit_opt()
        print('int8 conv_bn_fusion enabled')
        model.model = ipex.fx.conv_bn_fuse(model.model)

        if args.calibration:
            print("runing int8 LLGA calibration step\n")
            conf = ipex.AmpConf(torch.int8)
            for nbatch, (img, img_id, img_size, bbox, label) in enumerate(val_dataloader):
                with ipex.amp.calibrate(), torch.no_grad():
                    ploc, plabel, _ = model(img)
            conf.save(args.configure)

        else:
            conf = ipex.AmpConf(torch.int8, args.configure_dir)
            if args.dummy:
                print('runing int8 dummy inputs inference path')
                img = torch.randn(args.batch_size, 3, 1200, 1200).to(ipex.DEVICE)
                for nbatch in range(args.iteration):
                    with torch.no_grad():
                        with ipex.AutoMixPrecision(conf, running_mode="inference"):
                            if nbatch >= args.warmup_iterations:
                                start_time=time.time()
                            ploc, plabel,_ = model(img)
                            if nbatch >= args.warmup_iterations:
                                inference_time.update(time.time() - start_time)
                            if nbatch % args.print_freq == 0:
                                progress.display(nbatch)
            else:
                print('runing int8 real inputs inference path')
                for nbatch, (img, img_id, img_size, bbox, label) in enumerate(val_dataloader):
                    with torch.no_grad():
                        with ipex.AutoMixPrecision(conf, running_mode="inference"):
                            inp = img.to(ipex.DEVICE)
                            if nbatch >= args.warmup_iterations:
                                start_time=time.time()
                            ploc, plabel,_ = model(inp)
                            if nbatch >= args.warmup_iterations:
                                inference_time.update(time.time() - start_time)
                                end_time = time.time()
                            try:
                                results = encoder.decode_batch(ploc, plabel, 0.50, 200,device=device)
                            except:
                                print("No object detected in idx: {}".format(idx))
                                continue
                            if nbatch >= args.warmup_iterations:
                                decoding_time.update(time.time() - end_time)
                            (htot, wtot) = [d.cpu().numpy() for d in img_size]
                            img_id = img_id.cpu().numpy()
                            # Iterate over batch elements
                            for img_id_, wtot_, htot_, result in zip(img_id, wtot, htot, results):
                                loc, label, prob = [r.cpu().numpy() for r in result]
                                # Iterate over image detections
                                for loc_, label_, prob_ in zip(loc, label, prob):
                                    ret.append([img_id_, loc_[0]*wtot_, \
                                                loc_[1]*htot_,
                                                (loc_[2] - loc_[0])*wtot_,
                                                (loc_[3] - loc_[1])*htot_,
                                                prob_,
                                                inv_map[label_]])

                            if nbatch % args.print_freq == 0:
                                progress.display(nbatch)
                            if nbatch == args.iteration:
                                break
    else:
        if args.dummy:
            print('runing fp32 dummy inputs inference path')
            img = torch.randn(args.batch_size, 3, 1200, 1200)
            if use_cuda:
                img = img.to('cuda')
            elif args.ipex:
                img = img.to(ipex.DEVICE)
            enable_autocast = False
            if args.autocast:
                print('autocast enabled')
                enable_autocast = True
            else:
                print('autocast disabled')
                enable_autocast = False
            with ipex.amp.autocast(enabled=enable_autocast, configure=ipex.conf.AmpConf(torch.bfloat16)):
                for nbatch in range(args.iteration):
                    with torch.no_grad():
                        if nbatch >= args.warmup_iterations:
                            start_time=time.time()
                        if nbatch == 5:
                            print("Profilling")
                            with torch.autograd.profiler.profile(use_cuda=False, record_shapes=True) as prof:
                                ploc, plabel,_ = model(img)
                            print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                            prof.export_chrome_trace("torch_throughput.json")
                        else:
                            ploc, plabel,_ = model(img)
                        if nbatch >= args.warmup_iterations:
                            inference_time.update(time.time() - start_time)
                        if nbatch % args.print_freq == 0:
                            progress.display(nbatch)
        else:
            print('runing real inputs path')
            if args.autocast:
                print('bf16 autocast enabled')
                print('bf16 conv_bn_fusion enabled')
                model.model = ipex.fx.conv_bn_fuse(model.model)
                print('enable nhwc')
                model = model.to(memory_format=torch.channels_last)
                if args.jit:
                    print('enable jit')
                    #model = ipex.optimize(model, dtype=torch.bfloat16, level="O1")
                    with ipex.amp.autocast(enabled=True, configure=ipex.conf.AmpConf(torch.bfloat16)), torch.no_grad(): 
                        model = torch.jit.trace(model, torch.randn(args.batch_size, 3, 1200, 1200).to(memory_format=torch.channels_last)).eval()
                    model = torch.jit.freeze(model)
                    for nbatch, (img, img_id, img_size, bbox, label) in enumerate(val_dataloader):
                        print("nbatch: {}".format(nbatch))
                        with torch.no_grad():
                            if use_cuda:
                                img = img.to('cuda')
                            elif args.ipex:
                                img = img.to(ipex.DEVICE)

                            if nbatch >= args.warmup_iterations:
                                start_time=time.time()
                            
                            img = img.to(memory_format=torch.channels_last)
                            if args.profile and nbatch == 99:
                                print("Profilling")
                                with torch.autograd.profiler.profile(use_cuda=False, record_shapes=True) as prof:
                                    ploc, plabel = model(img)
                                print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                                prof.export_chrome_trace("torch_throughput.json")
                            else:
                                ploc, plabel = model(img)
                            if nbatch >= args.warmup_iterations:
                                inference_time.update(time.time() - start_time)
                                end_time = time.time()

                            try:
                                if args.profile and nbatch == 49:
                                    with torch.autograd.profiler.profile(use_cuda=False, record_shapes=True) as prof:
                                       results = encoder.decode_batch(ploc, plabel, 0.50, 200,device=device)
                                    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                                    prof.export_chrome_trace("torch_decode_throughput.json")
                                else:
                                    results = encoder.decode_batch(ploc, plabel, 0.50, 200,device=device)
                            except:
                                print("No object detected in idx: {}".format(idx))
                                continue
                            if nbatch >= args.warmup_iterations:
                                decoding_time.update(time.time() - end_time)
                            (htot, wtot) = [d.cpu().numpy() for d in img_size]
                            img_id = img_id.cpu().numpy()

                            for img_id_, wtot_, htot_, result in zip(img_id, wtot, htot, results):
                                loc, label, prob = [r.cpu().numpy() for r in result]
                                # Iterate over image detections
                                for loc_, label_, prob_ in zip(loc, label, prob):
                                    ret.append([img_id_, loc_[0]*wtot_, \
                                                loc_[1]*htot_,
                                                (loc_[2] - loc_[0])*wtot_,
                                                (loc_[3] - loc_[1])*htot_,
                                                prob_,
                                                inv_map[label_]])

                            if nbatch % args.print_freq == 0:
                                progress.display(nbatch)
                            if nbatch == args.iteration:
                                break 
                else:
                    print('autocast imperative path')
                    with ipex.amp.autocast(enabled=True, configure=ipex.conf.AmpConf(torch.bfloat16)):
                        for nbatch, (img, img_id, img_size, bbox, label) in enumerate(val_dataloader):
                            #print("nbatch: {}".format(nbatch))
                            with torch.no_grad():
                                if use_cuda:
                                    img = img.to('cuda')
                                elif args.ipex:
                                    img = img.to(ipex.DEVICE)

                                if nbatch >= args.warmup_iterations:
                                    start_time=time.time()
                                #ploc, plabel = model(img)
                                img = img.contiguous(memory_format=torch.channels_last)

                                if args.profile and nbatch == 100:
                                    print("Profilling")
                                    with torch.autograd.profiler.profile(use_cuda=False, record_shapes=True) as prof:
                                        ploc, plabel = model(img)
                                    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                                    prof.export_chrome_trace("torch_throughput.json")
                                else:
                                    ploc, plabel = model(img)

                                if nbatch >= args.warmup_iterations:
                                    inference_time.update(time.time() - start_time)
                                    end_time = time.time()
                                
                                with ipex.amp.autocast(enabled=False):
                                    try:
                                        #results = encoder.decode_batch(ploc.to_dense().to(torch.float32), plabel.to_dense().to(torch.float32), 0.50, 200,device=device)
                                        results = encoder.decode_batch(ploc, plabel, 0.50, 200, device=device)
                                    except:
                                        print("No object detected in idx: {}".format(idx))
                                        continue
                                    if nbatch >= args.warmup_iterations:
                                        decoding_time.update(time.time() - end_time)
                                    (htot, wtot) = [d.cpu().numpy() for d in img_size]
                                    img_id = img_id.cpu().numpy()

                                    for img_id_, wtot_, htot_, result in zip(img_id, wtot, htot, results):
                                        loc, label, prob = [r.cpu().numpy() for r in result]
                                        # Iterate over image detections
                                        for loc_, label_, prob_ in zip(loc, label, prob):
                                            ret.append([img_id_, loc_[0]*wtot_, \
                                                        loc_[1]*htot_,
                                                        (loc_[2] - loc_[0])*wtot_,
                                                        (loc_[3] - loc_[1])*htot_,
                                                        prob_,
                                                        inv_map[label_]])

                                    if nbatch % args.print_freq == 0:
                                        progress.display(nbatch)
                                    if nbatch == args.iteration:
                                        break
            else:
                print('autocast disabled, fp32 is used')
                print('fp32 conv_bn_fusion enabled')
                model.model = ipex.fx.conv_bn_fuse(model.model)
                print('enable nhwc')
                model = model.to(memory_format=torch.channels_last)
                if args.jit:
                    print("enable jit")
                    with torch.no_grad():
                        model = torch.jit.trace(model, torch.randn(args.batch_size, 3, 1200, 1200).to(memory_format=torch.channels_last))
                        #model = torch.jit.trace(model, torch.randn(args.batch_size, 3, 1200, 1200))
                for nbatch, (img, img_id, img_size, bbox, label) in enumerate(val_dataloader):
                    with torch.no_grad():
                        if use_cuda:
                            img = img.to('cuda')
                        elif args.ipex:
                            img = img.to(ipex.DEVICE)

                        if nbatch >= args.warmup_iterations:
                            start_time=time.time()
                        
                        img = img.contiguous(memory_format=torch.channels_last)
                        if args.profile and nbatch == 100:
                            print("Profilling")
                            with torch.autograd.profiler.profile(use_cuda=False, record_shapes=True) as prof:
                                ploc, plabel = model(img)
                            print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                            prof.export_chrome_trace("torch_throughput.json")
                        else:
                            ploc, plabel = model(img)
                        if nbatch >= args.warmup_iterations:
                            inference_time.update(time.time() - start_time)
                            end_time = time.time()
                        try:
                            if args.profile and nbatch == 49:
                                with torch.autograd.profiler.profile(use_cuda=False, record_shapes=True) as prof:
                                    results = encoder.decode_batch(ploc, plabel, 0.50, 200,device=device)
                                print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                                prof.export_chrome_trace("torch_fp32_decode_throughput.json")
                            else:
                                results = encoder.decode_batch(ploc, plabel, 0.50, 200,device=device)
                            #results = encoder.decode_batch(ploc, plabel, 0.50, 200,device=device)
                        except:
                            print("No object detected in idx: {}".format(idx))
                            continue
                        if nbatch >= args.warmup_iterations:
                            decoding_time.update(time.time() - end_time)
                        (htot, wtot) = [d.cpu().numpy() for d in img_size]
                        img_id = img_id.cpu().numpy()
                        # Iterate over batch elements
                        for img_id_, wtot_, htot_, result in zip(img_id, wtot, htot, results):
                            loc, label, prob = [r.cpu().numpy() for r in result]
                            # Iterate over image detections
                            for loc_, label_, prob_ in zip(loc, label, prob):
                                ret.append([img_id_, loc_[0]*wtot_, \
                                            loc_[1]*htot_,
                                            (loc_[2] - loc_[0])*wtot_,
                                            (loc_[3] - loc_[1])*htot_,
                                            prob_,
                                            inv_map[label_]])

                        if nbatch % args.print_freq == 0:
                            progress.display(nbatch)
                        if nbatch == args.iteration:
                            break
    print("Predicting Ended, total time: {:.2f} s".format(time.time()-start))

    batch_size = args.batch_size
    latency = inference_time.avg / batch_size * 1000
    perf = batch_size / inference_time.avg
    print('inference latency %.2f ms'%latency)
    print('inference performance %.2f fps'%perf)

    if not args.dummy:
        latency = decoding_time.avg / batch_size * 1000
        perf = batch_size / decoding_time.avg
        print('decoding latency %.2f ms'%latency)
        print('decodingperformance %.2f fps'%perf)

        cocoDt = cocoGt.loadRes(np.array(ret))

        E = COCOeval(cocoGt, cocoDt, iouType='bbox')
        E.evaluate()
        E.accumulate()
        E.summarize()
        print("Current AP: {:.5f} AP goal: {:.5f}".format(E.stats[0], threshold))

        return (E.stats[0] >= threshold) #Average Precision  (AP) @[ IoU=050:0.95 | area=   all | maxDets=100 ]
    else:
        return False

def eval_ssd_r34_mlperf_coco(args):
    from coco import COCO
    # Check that GPUs are actually available
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    dboxes = dboxes_R34_coco(args.image_size, args.strides)

    encoder = Encoder(dboxes)

    val_trans = SSDTransformer(dboxes, (args.image_size[0], args.image_size[1]), val=True)

    if not args.dummy:
        val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
        val_coco_root = os.path.join(args.data, "val2017")

        cocoGt = COCO(annotation_file=val_annotate)
        val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
        inv_map = {v:k for k,v in val_coco.label_map.items()}

        val_dataloader = DataLoader(val_coco,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    sampler=None,
                                    num_workers=args.workers)
        labelnum = val_coco.labelnum
    else:
        cocoGt = None
        encoder = None
        inv_map = None
        val_dataloader = None
        labelnum = 81

    ssd_r34 = SSD_R34(labelnum, strides=args.strides)

    if args.checkpoint:
        print("loading model checkpoint", args.checkpoint)
        od = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
        ssd_r34.load_state_dict(od["model"])

    if use_cuda:
        ssd_r34.cuda(args.device)
    elif args.ipex:
        ssd_r34 = ssd_r34.to(ipex.DEVICE)
    #if args.jit:
        #ssd_r34 = torch.jit.script(ssd_r34)
    coco_eval(ssd_r34, val_dataloader, cocoGt, encoder, inv_map, args)

def main():
    args = parse_args()

    print(args)
    if not os.path.isdir('./models'):
        os.mkdir('./models')

    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed)
        np.random.seed(seed=args.seed)
    if not args.no_cuda:
        torch.cuda.set_device(args.device)
        torch.backends.cudnn.benchmark = True
    eval_ssd_r34_mlperf_coco(args)

if __name__ == "__main__":
    main()
