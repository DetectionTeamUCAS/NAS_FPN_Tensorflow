from data.lib_coco.PythonAPI.pycocotools.coco import COCO
from data.lib_coco.PythonAPI.pycocotools.cocoeval import COCOeval


def cocoval(detected_json, eval_json):
    eval_gt = COCO(eval_json)

    eval_dt = eval_gt.loadRes(detected_json)
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='bbox')

    # cocoEval.params.imgIds = eval_gt.getImgIds()
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


detected_json = '/home/yangxue/isilon/yangxue/code/ADAS/output/yangxue/fpn/fpn.res50.coco.roialign.2x.detectron.new.concat/eval_dump/epoch-2.coco'
eval_gt = '/unsullied/sharefs/_research_detection/GeneralDetection/COCO/data/MSCOCO/instances_minival2014.json'
cocoval(detected_json, eval_gt)