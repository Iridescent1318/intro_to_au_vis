clear all;clc
img_id = 2;
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

imshow(img)
hold on
draw_lines(lines, W, 12)
[vps, vps_class, m, b] = get_vps(lines, 12, H, W);
scatter(point_target(1), point_target(2), 'x')
draw_person_line(mask_person1);
draw_person_line(mask_person2);
draw_target_line(mask_ground, point_target(1:2));

figure
m_g = 32;
m_p1 = 32;
m_p2 = 32;
m_t = 32;
imshow(mask_ground*m_g + mask_person1*m_p1 + mask_person2*m_p2 + mask_target*m_t);

function [vps, vps_class, m, b] = get_vps(lines, max_lines_nums, H, W)
    vps = [];
    for i=1:min(length(lines), max_lines_nums)
        for j=i:min(length(lines), max_lines_nums)
            if(i == j)
                continue;
            end
            nvps = cross(lines(i, :), lines(j, :));
            nvps = nvps / nvps(3);
            vps = [vps;nvps(1:2)];
        end
    end
    vps = vps';
    to_del = zeros(1, length(vps(1, :)));
    for i=1:length(to_del)
        if(vps(1, i) > 0 && vps(1, i) < W && vps(2, i) > 0 && vps(2, i) < H)
            to_del(i) = 1;
        end
    end
    vps(:, find(to_del == 1)) = [];
    color = [[1,0,0];[0,1,0];[0,0,1];[1,1,0];[1,0,1];[0,1,1]];
    vps_class = kmeans(vps', 2);
    scatter(vps(1, :), vps(2, :), 144, color(vps_class, :), '.')
    
    nearest_dis = zeros(1, length(vps(1, :)));
    for i=1:length(nearest_dis)
         nearest_d = Inf;
        for j=1:length(nearest_dis)
            if(i == j)
                continue;
            end
            if((vps(1, i)-vps(1, j))^2 + (vps(2, i)-vps(2, j))^2 < nearest_d)
                nearest_d = (vps(1, i)-vps(1, j))^2 + (vps(2, i)-vps(2, j))^2;
            end
        end
        nearest_dis(i)= nearest_d;
    end  
    nearest_dis = 1 ./ nearest_dis;

    partial_k_k = sum( nearest_dis.*vps(1,:).^2 );
    partial_k_b = sum( nearest_dis.*vps(1,:) );
    partial_k_c = sum( nearest_dis.*vps(1,:).*vps(2,:) );
    partial_b_k = partial_k_b;
    partial_b_b = sum( nearest_dis );
    partial_b_c = sum( nearest_dis.*vps(2,:) );
    A = [partial_k_k, partial_k_b;partial_b_k, partial_b_b];
    B = [partial_k_c; partial_b_c];
    C = A\B;
    m = C(1);
    b = C(2);
    x0 = -2000:3000;
    y0 = m.*x0+b;
    plot(x0, y0, 'Color', 'c', 'LineWidth', 3);
end

function draw_lines(lines, W, max_lines_nums)
    k = -lines(:, 1)./lines(:, 2);
    b = -lines(:, 3)./lines(:, 2);
    c_index = kmeans(k, 2);
    for i=1:min(length(lines), max_lines_nums)
        x = -2.5*W:2.5*W;
        y = k(i).*x+b(i);
        color = ['r', 'g', 'b', 'c', 'm', 'y'];
        c = c_index(i);
        plot(x, y, 'Color', color(c), 'LineWidth', 1)
    end
end

function draw_person_line(person_mask)
    plist = ones(sum(sum(person_mask)), 2);
    plist_k = 1;
    for i=1:length(person_mask(1, :))
        for j=1:length(person_mask(:, 1))
            if(person_mask(j, i) == 1)
                plist(plist_k, :) = [i, j];
                plist_k = plist_k+1;
            end
        end
    end
    coeff = pca(plist);
    k = coeff(2, 2) / coeff(1, 2);
    y = -k.*plist(:, 1)+plist(:, 2);
    y_max = -Inf;
    y_min = Inf;
    for i=1:length(y)
        if(y(i) >= y_max)
            y_max = y(i);
            y_max_point = plist(i, :);
        end
        if(y(i) <= y_min)
            y_min = y(i);
            y_min_point = plist(i, :);
        end
    end
    person_y = linspace(min(y_max_point(2), y_min_point(2)), max(y_max_point(2), y_min_point(2)), 100);
    person_x = -k .* (person_y - y_min_point(2)) + y_min_point(1);    
    plot(person_x, person_y, 'Color', 'y', 'LineWidth', 2);
end

function draw_target_line(ground_mask, target_point)
    d_min = Inf;
    d_min_point = [0, 0];
    for i=1:length(ground_mask(1, :))
        for j=1:length(ground_mask(:, 1))
            if(ground_mask(j, i) == 1)
                if((target_point(1)-i)^2 + (target_point(2)-j)^2 < d_min)
                    d_min = (target_point(1)-i).^2 + (target_point(2)-j).^2;
                    d_min_point = [i, j];
                end
            end
        end
    end
    target_x = linspace( min(d_min_point(1), target_point(1)), max(d_min_point(1), target_point(1)), 100);
    slope = (target_point(2) - d_min_point(2)) / (target_point(1) - d_min_point(1));
    target_y = slope.*(target_x - target_point(1)) + target_point(2);
    plot(target_x, target_y, 'Color', 'm', 'LineWidth', 1.5);
end
