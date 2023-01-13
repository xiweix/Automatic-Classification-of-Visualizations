# grad-CAM and grad-CAM++, part of the code is from:
# https://github.com/1Konny/gradcam_plus_plus-pytorch/blob/master/gradcam.py

import os
import warnings
import numpy as np
import csv
import PIL
import click
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from gradcam_utils import visualize_cam, Normalize
from gradcam import GradCAM, GradCAMpp

warnings.filterwarnings("ignore")


@click.command()
@click.option('--comp-gradcam',
              default=True,
              help='whether compare gradcam results')
@click.option('--comp-gradcampp',
              default=True,
              help='whether compare gradcam++ results')
@click.option('--model-list',
              type=list,
              default=['googlenet'],
              help='model list to be visualized')
def main(model_list, comp_gradcam, comp_gradcampp):
    base_dir = os.path.join(os.path.dirname(os.getcwd()), 'beagle_output', 'random_model.googlenet-batchsize.128')
    min_loss_dir = os.path.join(base_dir, 'test-minloss_result', 'test_info')
    os.makedirs(min_loss_dir, exist_ok=True)
    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    num_classes = 24
    # Load torchvision models and make model dictionaries
    cam_dict = dict()
    if 'googlenet' in model_list:
        googlenet = getattr(models, 'googlenet')(num_classes=num_classes)
        googlenet.eval()
        if use_cuda == True:
            googlenet.cuda()
        googlenet_model_dict = dict(type='googlenet', arch=googlenet,layer_name='inception5b', input_size=(224, 224))
        googlenet_gradcam = GradCAM(googlenet_model_dict, True)
        googlenet_gradcampp = GradCAMpp(googlenet_model_dict, True)
        cam_dict['googlenet'] = [googlenet_gradcam, googlenet_gradcampp]

    out_label = np.load(os.path.join(base_dir, 'test-minloss_result', 'Epoch169_labels.npy'))
    out_pred = np.load(os.path.join(base_dir, 'test-minloss_result', 'Epoch169_predictions.npy'))
    correct_idx = []
    correct_label = []
    wrong_idx = []
    wrong_label_rel = []
    wrong_label_prd = []
    print(out_label.shape[0])
    for i in range(out_label.shape[0]):
        if out_label[i] == out_pred[i]:
            correct_idx.append(i)
            correct_label.append(out_label[i])
        else:
            wrong_idx.append(i)
            wrong_label_rel.append(out_label[i])
            wrong_label_prd.append(out_pred[i])
    print('correct')
    print(len(correct_idx))
    print(len(correct_label))
    print('wrong')
    print(len(wrong_idx))
    print(len(wrong_label_rel))
    print(len(wrong_label_prd))

    csv_file = os.path.join(os.path.dirname(os.getcwd()), 'dataset', 'test.csv')
    correct_img_name_list = []
    correct_img_folder_list = []
    wrong_img_name_list = []
    wrong_img_folder_list = []
    with open(csv_file, 'r') as f:
        prj_info = csv.reader(f)
        for row in prj_info:
            if prj_info.line_num == 1:
                print('start')
                # with open(os.path.join(min_loss_dir, f'correct_info.csv'), 'a') as fcsv:
                #     writer = csv.writer(fcsv)
                #     writer.writerow(row)
                # with open(os.path.join(min_loss_dir, f'wrong_info.csv'), 'a') as fcsv:
                #     writer = csv.writer(fcsv)
                #     writer.writerow(row)
            else:
                index = prj_info.line_num - 2
                if index in correct_idx:
                    correct_img_name_list.append(row[0])
                    correct_img_folder_list.append(row[5])
                    # with open(os.path.join(min_loss_dir, f'correct_info.csv'), 'a') as fcsv:
                    #     writer = csv.writer(fcsv)
                    #     writer.writerow(row)
                elif index in wrong_idx:
                    wrong_img_name_list.append(row[0])
                    wrong_img_folder_list.append(row[5])
                    # with open(os.path.join(min_loss_dir, f'wrong_info.csv'), 'a') as fcsv:
                    #     writer = csv.writer(fcsv)
                    #     writer.writerow(row)

    print(f'correct: {len(correct_img_name_list)} , {len(correct_img_folder_list)} images to be visualized')
    print(f'wrong: {len(wrong_img_name_list)} , {len(wrong_img_folder_list)} images to be visualized')

    # np.save(os.path.join(min_loss_dir, 'correct_idx.npy'), correct_idx)
    # np.save(os.path.join(min_loss_dir, 'correct_label.npy'), correct_label)
    # np.save(os.path.join(min_loss_dir, 'correct_img_name_list.npy'), correct_img_name_list)
    # np.save(os.path.join(min_loss_dir, 'correct_img_folder_list.npy'), correct_img_folder_list)


    # np.save(os.path.join(min_loss_dir, 'wrong_idx.npy'), wrong_idx)
    # np.save(os.path.join(min_loss_dir, 'wrong_label_rel.npy'), wrong_label_rel)
    # np.save(os.path.join(min_loss_dir, 'wrong_label_prd.npy'), wrong_label_prd)
    # np.save(os.path.join(min_loss_dir, 'wrong_img_name_list.npy'), wrong_img_name_list)
    # np.save(os.path.join(min_loss_dir, 'wrong_img_folder_list.npy'), wrong_img_folder_list)

    for i in range(len(wrong_img_name_list)):
        img_name = wrong_img_name_list[i]
        img_folder = wrong_img_folder_list[i]
        print(img_name)
        # Load image
        img_dir = os.path.join(os.path.dirname(os.getcwd()), 'dataset')
        img_path = os.path.join(img_dir, img_folder, 'images', img_name)
        pil_img = PIL.Image.open(img_path).convert('RGB')

        # define the output direction
        base_name = os.path.splitext(img_name)[0]
        output_dir = os.path.join(base_dir, 'wrong_vis_results', img_folder)
        os.makedirs(output_dir, exist_ok=True)

        # preprocess image
        # normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[
        #                     0.229, 0.224, 0.225])
        torch_img = torch.from_numpy(np.asarray(pil_img)).permute(
            2, 0, 1).unsqueeze(0).float().div(255)
        if use_cuda == True:
            torch_img = torch_img.cuda()
        torch_img = F.upsample(torch_img, size=(224, 224),
                            mode='bilinear', align_corners=False)
        # normed_torch_img = normalizer(torch_img)


        # Feedforward image, calculate GradCAM/GradCAM++, and gather results
        # for gradcam, gradcam_pp in cam_dict.values():
        i = 0
        for gradcam, gradcam_pp in cam_dict.values():
            gradcam_images = []
            gradcampp_images = []
            com_images = []

            mask, _ = gradcam(torch_img)
            heatmap, result = visualize_cam(mask, torch_img)

            mask_pp, _ = gradcam_pp(torch_img)
            heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)

            gradcam_images.append(torch.stack(
                [torch_img.squeeze().cpu(), heatmap, result], 0))
            gradcampp_images.append(torch.stack(
                [torch_img.squeeze().cpu(), heatmap_pp, result_pp], 0))
            com_images.append(torch.stack([torch_img.squeeze().cpu(), heatmap, result, heatmap_pp, result_pp], 0))

            gradcam_images = make_grid(torch.cat(gradcam_images, 0), nrow=3)
            gradcampp_images = make_grid(torch.cat(gradcampp_images, 0), nrow=3)
            compare_images = make_grid(torch.cat(com_images, 0), nrow=5)

            # Save and show results
            save_image(heatmap, os.path.join(
                output_dir, f'{base_name}_gradcam_heatmap.png'))
            save_image(result, os.path.join(
                output_dir, f'{base_name}_gradcam_heatmap_w_img.png'))
            save_image(gradcam_images, os.path.join(
                output_dir, f'{base_name}_gradcam.png'))

            save_image(heatmap_pp, os.path.join(
                output_dir, f'{base_name}_gradcampp_heatmap.png'))
            save_image(result_pp, os.path.join(
                output_dir, f'{base_name}_gradcampp_heatmap_w_img.png'))
            save_image(gradcampp_images, os.path.join(
                output_dir, f'{base_name}_gradcampp.png'))

            save_image(compare_images, os.path.join(
                output_dir, f'{base_name}_compare.png'))
            i += 1


if __name__ == '__main__':
    main()
