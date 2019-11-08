function H = HeightEstimator(img,mask_ground,mask_target,mask_person1,mask_person2,ref_height1,ref_height2,point_target,lines)
    imshow(img)
    [H,W,~] = size(img);
    max_lines_nums = 8;
    hold on
    draw_lines(lines, W, max_lines_nums)
    vph = get_vps(lines, max_lines_nums, H, W);
    scatter(point_target(1), point_target(2), 'x')
    p1_xy = draw_person_line(mask_person1);
    p2_xy = draw_person_line(mask_person2);
    t_xy = draw_target_line(mask_ground, point_target(1:2));

    p1_line = cross(p1_xy(:, 1), p1_xy(:, 2));
    p1_line = p1_line / p1_line(3);
    p2_line = cross(p2_xy(:, 1), p2_xy(:, 2));
    p2_line = p2_line / p2_line(3);
    t_line = cross(t_xy(:, 1), t_xy(:, 2));
    t_line = t_line / t_line(3);

    vpv1 = estimate_vpointv(p1_line, t_line);
    vpv2 = estimate_vpointv(p2_line, t_line);
    vpv = (vpv1+vpv2)/2;

    [p1_t, p1_b] = select_top_bottom(p1_xy);
    [p2_t, p2_b] = select_top_bottom(p2_xy);
    [t_t, t_b] = select_top_bottom(t_xy);

    H1 = estimate_height(vpv, vph, p1_t, p1_b, ref_height1, t_t, t_b);
    H2 = estimate_height(vpv, vph, p2_t, p2_b, ref_height2, t_t, t_b);
    H0 = (H1 + H2)/2

    function vph = get_vps(lines, max_lines_nums, H, W)
        vps = [];
        alpha = 1.65;
        line_num = min(length(lines(:, 1)), max_lines_nums);
        for i=1:line_num
            for j=i:line_num
                if(i == j)
                    continue;
                end
                nvps = cross(lines(i, :), lines(j, :));
                nvps = nvps / nvps(3);
                vps = [vps;nvps];
            end
        end

        [~, score, ~] = pca(vps(:, 1:2));
        [~, s_sort_idx] = sort(score(:, 2));
        vps_num = 0;
        if(mod(line_num, 2) == 0)
            vps_num = line_num^2/4-line_num/2;
        else
            vps_num = (line_num-1)^2/4-1;
        end

    %     vps_color = zeros(1, length(vps(:, 1)));
    %     vps_color(s_sort_idx(1:vps_num)) = 65535;
    %     figure
    %     scatter(score(:, 1), score(:, 2), 30, vps_color);

        vps = vps(s_sort_idx(1:vps_num), 1:3);
        vps_class = kmeans(vps(:, 1:2), 2);
        vps_class_idx_1 = find(vps_class == 1);
        vps_class_idx_2 = find(vps_class == 2);

        l_bs = [];
        for i=1:length(vps_class_idx_1)
            for j=1:length(vps_class_idx_2)
                newline = cross(vps(vps_class_idx_1(i), :), vps(vps_class_idx_2(j), :));
                newline = newline / newline(3);
                nl_b = -1/newline(2);
                l_bs = [l_bs, nl_b];
            end
        end
        sigma = std(l_bs);

        d_weight = ones(length(vps(:, 1)), 1);

        vps_x = vps(:, 1);
        vps_y = vps(:, 2);
        partial_k_k = sum( d_weight.*vps_x.^2 );
        partial_k_b = sum( d_weight.*vps_x );
        partial_k_c = sum( d_weight.*vps_x.*vps_y );
        partial_b_k = partial_k_b;
        partial_b_b = sum( d_weight );
        partial_b_c = sum( d_weight.*vps_y );
        A = [partial_k_k, partial_k_b;partial_b_k, partial_b_b];
        B = [partial_k_c; partial_b_c];
        C = A\B;
        m = C(1);
        b = C(2);

        op_m = m;
        op_b = b + sigma * alpha;

        x0 = -3000:3000;
        y0 = op_m.*x0+op_b;
        vph = [op_m/op_b, -1/op_b, 1];
        plot(x0, y0, 'Color', 'g', 'LineWidth', 3);
    end

    function draw_lines(lines, W, max_lines_nums)
        k = -lines(:, 1)./lines(:, 2);
        b = -lines(:, 3)./lines(:, 2);
        for i=1:min(length(lines), max_lines_nums)
            x = -2.5*W:2.5*W;
            y = k(i).*x+b(i);
            plot(x, y, 'Color', 'r', 'LineWidth', 1)
        end
    end

    function person_xy = draw_person_line(person_mask)
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
        person_xy = [person_x(1), person_x(end); person_y(1), person_y(end); 1, 1];
        plot(person_x, person_y, 'Color', 'y', 'LineWidth', 2);
    end

    function target_xy = draw_target_line(ground_mask, target_point)
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
        target_xy = [target_x(1), target_x(end); target_y(1), target_y(end); 1, 1];
        plot(target_x, target_y, 'Color', 'b', 'LineWidth', 1.5);
    end

    function vpv = estimate_vpointv(line1, line2)
        vpv = cross(line1, line2);
        vpv = vpv / vpv(3);
        scatter(vpv(1), vpv(2), 'p')
    end

    function [top, bottom] = select_top_bottom(xy)
        if(xy(2, 1) > xy(2, 2))
            top = xy(:, 2);
            bottom = xy(:, 1);
        else
            top = xy(:, 1);
            bottom = xy(:, 2);
        end
    end

    function target_height = estimate_height(vpoint_vertical, vline, ref_top, ref_bottom, ref_height, target_top, target_bottom)
        % all inputs are in (x, y, 1) form
        b_r = cross(ref_bottom, ref_top);
        b_r = b_r / b_r(3);
        v_r = cross(vpoint_vertical, ref_top);
        v_r = v_r / v_r(3);
        s_factor = -norm(b_r) / (ref_height * (vline * ref_bottom) * norm(v_r));
        b0_t0 = cross(target_bottom, target_top);
        b0_t0 = b0_t0 / b0_t0(3);
        v_t0 = cross(vpoint_vertical, target_top);
        v_t0 = v_t0 / v_t0(3);
        target_height = -norm(b0_t0) / (s_factor * (vline * target_bottom) * norm(v_t0));
    end
end