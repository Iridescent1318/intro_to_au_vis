clear all;clc
img_id = 3;
img = imread(['./hw/',num2str(img_id),'.jpg']);
img = imrotate(img,270);
f = fileread(['./hw/',num2str(img_id),'.json']);
j = jsondecode(f);
l_num = length(j.shapes)-4;
[H,W,~] = size(img);
mask_ground = uint8(poly2mask(j.shapes(1).points(:,1),j.shapes(1).points(:,2),H,W));
mask_target = uint8(poly2mask(j.shapes(2).points(:,1),j.shapes(2).points(:,2),H,W));
mask_person1 = uint8(poly2mask(j.shapes(3).points(:,1),j.shapes(3).points(:,2),H,W));
mask_person2 = uint8(poly2mask(j.shapes(4).points(:,1),j.shapes(4).points(:,2),H,W));
point_target = [j.point_target',1.0];

lines = [];
for i=1:l_num
    line = cross([j.shapes(4+i).points(1,:),1.0],[j.shapes(4+i).points(2,:),1.0]);
    line = line/line(3);
    lines = [lines;line];
end

ref_height1 = j.ref_height1;
ref_height2 = j.ref_height2;


%height: the estimated height of point_target in real world
%img:RGB image
%mask_ground: binary mask for the reference plain
%mask_target: binaray mask for the target architecture
%mask_person1: binary mask for the reference person1
%mask_person2: binary mask for the reference person2
%ref_height1: the height of reference person1 in real world
%ref_height2: the height of reference person2 in real world
%point_target: the target point whose height is to be estimated
%lines: several lines in the reference plain

%What you should do is to accomplish the following function
height = HeightEstimator(img,mask_ground,mask_target,mask_person1,mask_person2,ref_height1,ref_height2,point_target,lines);
height2 = MyHeightEstimator(img,mask_ground,mask_target,mask_person1,mask_person2,ref_height1,ref_height2,point_target,lines);
% figure
% imshow(mask_ground*50+mask_target*50+mask_person1*50+mask_person2*50)