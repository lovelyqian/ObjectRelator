import json
import os
from PIL import Image
import numpy as np
from pycocotools.mask import encode, decode, frPyObjects
from tqdm import tqdm
import copy
from natsort import natsorted
import cv2

if __name__ == '__main__':
    root_path = '/data/work2-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap'
    # 实验需改动
    save_path = os.path.join(root_path, 'egoexo_val_framelevel_violin_1113.json')

    # 获取takes_id
    split_path = "/home/yuqian_fu/Projects/ego-exo4d-relation/correspondence/SegSwap/data/split.json"
    with open(split_path, "r") as fp:
        data_split = json.load(fp)
    # val_set = data_split["train"]
    val_set = ["2fe390a8-1506-4420-9008-74199f92797b"]
    out = ['f2f93854-2634-449c-b68e-aebf4743ac9f', '7d59164c-e0bc-4ae0-95c9-733e4c8b0d6a',
           'cace80a1-42df-4cf4-a1ec-80647638a443', '8fa671be-2624-4783-8572-5f4b7722b6c0',
           'ca1434ea-b787-44ad-a9da-e0f7d5167a35', 'cfd2c825-45d1-4e59-b33f-b6dff8c174c8',
           '39d48b6a-66e8-4bbb-a596-4461b601cabc', 'f4dead01-fa3d-4aa5-8b59-13a0d9186dd2',
           '4cf43506-d0a6-4c42-9136-adb2ecd57411', '89815623-8ece-4e3c-8879-f1f32b299527',
           '5fd383f1-c8bb-42d7-b98b-7418d99d9bb4', 'b1b794e8-7839-46ab-b05f-f4b1c16d5420',
           'b7dbb47c-d850-4853-b434-7b20519ea9e5', '636eaa0b-d65d-4b25-bbdd-1065f84ef89e',
           'd2b0ee95-2a76-4b69-aebf-3c7e553f8e2b', 'cea1b20b-6e18-4bb6-87e0-164a2b8c3dc0',
           'fba2c124-99ec-40ed-8d6a-46808afe6d98', '549e5b97-f93a-4500-8f02-5be13017dce5',
           '6cbfa460-72a7-4038-b6af-4305a1cc05dd', '319a9983-f70c-4224-a3a7-33338c8a9f35',
           'bcfb6839-3c09-4d42-9d9c-59042f6ab721', '3ca2798d-cfe3-46c6-a8bc-cc4689bd6d75',
           'aa40670e-4487-4f60-a27b-26f7372ef8e7', '389cfa3f-3a4c-4b8f-9535-d7c95ffd594c',
           '214bdc0d-fa50-4a84-b771-c0a7bdeafadc', '08504348-4f72-477e-a08f-1050204ae55e',
           'd35a162e-c38e-4017-94d1-539f26651115', '925ebe22-f97a-4e79-aed3-9873cf461c3c',
           'c0b7d130-2004-4450-ad94-ea3167bd9fab', 'f1ce9be1-b623-4ce4-84e2-37e9f88eea86',
           'c288cb15-81c7-4463-814f-959c12740499', 'b69de073-c157-4cff-8eba-97bfc7baa012',
           '289c4873-5bf9-41b9-b784-aba52b54cd4d', '0bcb8b46-cc45-4bcb-a627-85633e54e060',
           '819780c3-97c0-4ac6-b37f-8c66abf8167d', '9512a137-40b6-49c6-a03d-a3340b9dd277',
           '393e9b60-504c-4cc7-a90b-eb78dd62d5ff', 'fb09baec-1a5a-40b4-8d72-e581c93fbd77',
           '3042b472-99cc-4407-a79e-f76916b95737', '4939903a-2c73-4633-ae15-618be39990e7',
           'ced0e340-b958-4505-badc-c8c2f256c145', 'e53ae33b-61b6-4c3e-8be0-5696f961704b',
           '759aa03c-c8e1-4fcb-8817-85948100ed33', '31df8578-1fd0-4406-9008-900a88f7990a',
           '9506c70a-1639-42ea-bbfc-2b9c0f8c9394', 'e46f9a53-7625-4827-8b92-79c958d3524e',
           '3125dca7-b99e-4b2b-8844-2d912619b353', 'b65a60e0-f224-4b3d-bf78-ba44e12c4ac1',
           'c53daadd-09f0-485b-a51c-1d5679f5fb09', 'b378d186-0587-40ed-afdf-875e6dfb5876',
           '8000dfb2-accb-4bb8-abfb-cc2d677d0b2f', 'ff93ae48-0daa-418c-9d5c-b5b0b6d23efb',
           '6d0a7c80-ae8c-4673-8f70-c09fd6fbebe8', 'f0cd03a5-9cd8-4510-87c1-a5c493197b75',
           'c16175f5-f990-48a1-9dcc-6a385f108687', '816dd81e-93b1-433c-83ff-264ae404a3bf',
           'f8bed5fe-3e09-4885-9539-edb4d5b2279a', '601c9c61-fc2b-4ac2-b3c4-dda557c2563b',
           '56c35d79-acb1-47a1-8590-7e5cb2585ee5', 'd5193bae-a7f5-4e8a-9c96-09f557c7ea9d',
           '8d7646a3-ce7b-425d-b16e-9a63a1166576', '0b89efcd-59bf-4f0b-a81a-50dee0b79982',
           '9681b4c6-9713-4bb3-aa9a-7df7daa4e74d', '69fac17f-6527-493d-8dac-cd3bb61ce23e',
           '5ee00a17-171f-46a4-927b-3aa9d0fe176e', 'ce914bec-f8c1-46ef-ae28-f1ff030801d1']

    # 实验需改动
    # val_set = ["92b2221b-ae92-44f0-bb31-e2d27cb736d6"]

    # 用来计数
    new_img_id = 0

    # 用来存储json中的数据
    egoexo_dataset = []

    '''
    build_DAVIS.py的代码逻辑是先处理每个视频的第一帧，第一帧中的unique_instances、高宽等信息用于该视频下后续的每一帧。
    注意，unique_instances代表的是第一帧下像素的所有类别信息，如果该视频下后续的帧中有像素的类别不在unique_instances中，会报错
    '''

    # bad_case = []

    for val_name in tqdm(val_set):
        # 不同视角下两个相机的总路径
        # if val_name in out:
        #     continue

        vid_root_path = os.path.join(root_path, val_name)
        anno_path = os.path.join(vid_root_path, "annotation.json")
        with open(anno_path, 'r') as fp:
            annotations = json.load(fp)
        # 取出本take下的所有物体
        # objs = list(annotations["masks"].keys())
        objs = natsorted(list(annotations["masks"].keys()))
        # print("the total obj num are:", len(objs))
        # print(f"objs:{objs}")
        # 将物体名称映射为id "cook":1 从1开始，区别于背景
        # TODO看看这个要不要修改为以obj_ref中的物体为准
        coco_id_to_cont_id = {coco_id: cont_id + 1 for cont_id, coco_id in enumerate(objs)}

        # 区分相机
        valid_cams = os.listdir(vid_root_path)
        # 这一行必须加
        valid_cams.remove("annotation.json")

        # 给相机排序，方便取出01开头的相机，因为序号小的相机对应的物体更多
        valid_cams = natsorted(valid_cams)
        # print("valid_cams:", valid_cams)

        ego_cams = []
        exo_cams = []
        for vc in valid_cams:
            if 'aria' in vc:
                ego_cams.append(vc)
            else:
                exo_cams.append(vc)

        ego = ego_cams[0]
        exo = exo_cams[0]
        # print(ego, exo)

        # ego、exo相机路径
        vid_ego_path = os.path.join(vid_root_path, ego)
        # vid_exo_path = os.path.join(vid_root_path, exo)

        # setting为ego->exo，所以ego作为第一帧，即visual prompt
        # 取出第一帧ego图像的id
        # 获取帧的索引时，不能简单地通过os.listdir来获取，因为路径下有的图片是没有标注的，需要以注释文件的索引为准
        # 路径下图片的索引和注释文件中的索引的关系：图片名称里的subsample_idx是包含annotations里的idx的 即有的图片是没有对应的注释的，所以会出现索引报错
        ego_frames = natsorted(os.listdir(vid_ego_path))
        ego_frames = [int(f.split(".")[0]) for f in ego_frames]

        # 先选出两个摄像机下都出现的物体作为总的物体范围,然后再判断在该摄像机视角下每一帧中出现了哪些物体
        # 也可能出现objs_both_have为空的情况，这时候就需要更换exo摄像机
        objs_both_have = []
        for obj in objs:
            if ego in annotations["masks"][obj].keys() and exo in annotations["masks"][obj].keys():
                objs_both_have.append(obj)

        if len(exo_cams) > 1:
            for cam in exo_cams[1:]:
                objs_both_have_tmp = []
                for obj in objs:
                    if ego in annotations["masks"][obj].keys() and cam in annotations["masks"][obj].keys():
                        objs_both_have_tmp.append(obj)
                if len(objs_both_have_tmp) > len(objs_both_have):
                    exo = cam
                    objs_both_have = objs_both_have_tmp
        # 如果没有物体范围，跳过本take
        # print("objs_both_have num:", len(objs_both_have))
        if len(objs_both_have) == 0:
            # bad_case.append(val_name)
            continue

        # print(ego, exo)
        # 确定exo的最终相机后，再定义exo的路径
        vid_exo_path = os.path.join(vid_root_path, exo)
        print(f"vid_exo_path:{vid_exo_path}")
        exo_frames = natsorted(os.listdir(vid_exo_path))
        # exo_frames = [int(f.split(".")[0]) for f in exo_frames]
        # exo_frames是字符串形式 be like ['1' '2' '3']
        exo_frames = [f.split(".")[0] for f in exo_frames]

        # 获取ego注释文件中的所有索引，用于后续和exo的交叉
        # 取所有ego obj annotated_frames最长的作为基准帧数
        # 后续对exo的操作以基准帧为核心，而不是以物体为核心
        # 取ego视角下出现时间最长的物体对应的所有注释帧，作为基准帧
        obj_ref = objs_both_have[0]
        for obj in objs_both_have:
            if len(list(annotations["masks"][obj_ref][ego].keys())) < len(list(annotations["masks"][obj][ego].keys())):
                obj_ref = obj
        ego_anno_frames = natsorted(list(annotations["masks"][obj_ref][ego].keys()))
        # TODO给frames排个序
        frames = natsorted(np.intersect1d(ego_anno_frames, exo_frames))
        # print(f"frames:{frames}")

        # 开始处理exo相机下的每一帧
        # 看看索引从1开始还是从0开始
        for idx in frames:
            coco_format_annotations = []
            filename = f"{idx}.jpg"
            sample_img_path = os.path.join(vid_exo_path, filename)
            sample_img_relpath = os.path.relpath(sample_img_path, root_path)

            first_frame_img_path = os.path.join(vid_ego_path, filename)
            first_frame_img_relpath = os.path.relpath(first_frame_img_path, root_path)

            obj_list_ego = []
            for obj in objs_both_have:
                if idx in annotations["masks"][obj][ego].keys():
                    mask_ego = decode(annotations["masks"][obj][ego][idx])
                    area_new = mask_ego.sum().astype(float)
                    if area_new != 0:
                        obj_list_ego.append(obj)
            # print("total obj num in ego", len(obj_list_ego))
            if len(obj_list_ego) == 0:
                continue

            obj_list_ego_new = []
            for obj in obj_list_ego:
                # TODO看看segmentation中count和size的顺序影不影响使用
                segmentation_tmp = annotations["masks"][obj][ego][idx]
                binary_mask = decode(segmentation_tmp)
                # print("original binary_mask_shape:", binary_mask.shape)

                # 对解码后的mask进行缩放，使得可以匹配ego图像的大小
                h, w = binary_mask.shape
                binary_mask = cv2.resize(binary_mask, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
                # 这里计算的area是resize后的mask面积
                area = binary_mask.sum().astype(float)
                if area == 0:
                    # obj_list_ego.remove(obj)
                    continue
                segmentation = encode(np.asfortranarray(binary_mask))
                segmentation = {
                    'counts': segmentation['counts'].decode('ascii'),
                    'size': segmentation["size"],
                }
                obj_list_ego_new.append(obj)
                coco_format_annotations.append(
                    {
                        'segmentation': segmentation,
                        'area': area,
                        'category_id': float(coco_id_to_cont_id[obj]),
                    }
                )
            if len(obj_list_ego_new) == 0:
                continue

            obj_list_exo = []
            for obj in obj_list_ego_new:
                if idx in annotations["masks"][obj][exo].keys():
                    mask_exo = decode(annotations["masks"][obj][exo][idx])
                    area_exo = mask_exo.sum().astype(float)
                    if area_exo != 0:
                        obj_list_exo.append(obj)

            # 检查exo下每一帧的物体数量，也会碰到有的帧一个物体也没有，这种直接跳过
            # print("total obj num in exo", len(obj_list_exo))
            if len(obj_list_exo) == 0:
                continue

            height, width = annotations["masks"][obj_list_exo[0]][exo][idx]["size"]
            # print("original exo mask_shape:" ,height,width)
            image_info = {
                'file_name': sample_img_relpath,
                'height': height // 4,
                'width': width // 4,
            }

            anns = []


            obj_list_exo_new = []
            # print("obj_list_exo_ori", obj_list_exo)
            for obj in obj_list_exo:
                assert obj in obj_list_ego_new, 'Found new target not in the first frame'
                segmentation_tmp = annotations["masks"][obj][exo][idx]
                binary_mask = decode(segmentation_tmp)
                # print("original ego binary_mask_shape", binary_mask.shape)
                h, w = binary_mask.shape
                binary_mask = cv2.resize(binary_mask, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
                # print("binary_mask", binary_mask.shape)
                area = binary_mask.sum().astype(float)
                if area == 0:
                    # obj_list_exo.remove(obj)
                    continue
                segmentation = encode(np.asfortranarray(binary_mask))
                segmentation = {
                    'counts': segmentation['counts'].decode('ascii'),
                    'size': segmentation['size'],
                }
                obj_list_exo_new.append(obj)
                anns.append(
                    {
                        'segmentation': segmentation,
                        'area': area,
                        'category_id': float(coco_id_to_cont_id[obj]),
                    }
                )
            # print("obj_list_exo_after", obj_list_exo)
            # print("obj_list_exo_new", obj_list_exo_new)
            if len(obj_list_exo_new) == 0:
                continue

            # 统计本帧下物体对应的id，方便后续根据category_id调整first_frame_anns
            sample_unique_instances = [float(coco_id_to_cont_id[obj]) for obj in obj_list_exo_new]
            # check_ids = [ann['category_id'] for ann in anns]
            # print("check_ids", check_ids)
            # print("sample_unique_instances", sample_unique_instances)
            # 查看每一帧下有哪些物体
            # print(f"sample_unique_instances in {idx}:{sample_unique_instances}")
            

            # deepcopy的目的是，后续要根据本exo帧中物体的数量对参考帧的注释进行调整，防止修改原始注释
            first_frame_anns = copy.deepcopy(coco_format_annotations)
            # 考虑本帧物体的数量小于参考帧的情况，仅取出参考帧中本帧有的物体的注释；但是实际情况下，有可能本帧物体的数量会大于参考帧，这时候就需要调整统计本帧下有哪些物体时，总的物体范围
            if len(anns) < len(first_frame_anns):
                first_frame_anns = [ann for ann in first_frame_anns if ann['category_id'] in sample_unique_instances]
            # ego_ids = [ann['category_id'] for ann in first_frame_anns]
            # print("ego_ids", ego_ids)
            # print("len anns", len(anns))
            # print("len firsrt ann", len(first_frame_anns))
            assert len(anns) == len(first_frame_anns)
            sample = {
                'image': sample_img_relpath,
                'image_info': image_info,
                'anns': anns,
                'first_frame_image': first_frame_img_relpath,
                'first_frame_anns': first_frame_anns,
                'new_img_id': new_img_id,
                'video_name': val_name,
            }
            egoexo_dataset.append(sample)
            new_img_id += 1

    with open(save_path, 'w') as f:
        json.dump(egoexo_dataset, f)
    print(f'Save at {save_path}. Total sample: {len(egoexo_dataset)}')
