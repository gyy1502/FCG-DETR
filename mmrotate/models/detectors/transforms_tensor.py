
import torch
import numpy as np
import cv2

def obb2poly_tensor_le90(obboxes):
    """Convert oriented bounding boxes to polygons using PyTorch tensors.

    Args:
        obbs (Tensor): [x_ctr,y_ctr,w,h,angle] with shape (N, 5)

    Returns:
        polys (Tensor): [x0,y0,x1,y1,x2,y2,x3,y3] with shape (N, 8)
    """
    # 确保输入是张量
    if not isinstance(obboxes, torch.Tensor):
        obboxes = torch.tensor(obboxes, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 获取设备信息
    device = obboxes.device
    
    # 分离各个组件
    center = obboxes[:, :2]
    w = obboxes[:, 2:3]
    h = obboxes[:, 3:4]
    theta = obboxes[:, 4:5]

    
    # 计算三角函数值
    Cos = torch.cos(theta)
    Sin = torch.sin(theta)
    
    # 计算两个向量
    vector1 = torch.cat([w / 2 * Cos, w / 2 * Sin], dim=-1)
    vector2 = torch.cat([-h / 2 * Sin, h / 2 * Cos], dim=-1)
    
    # 计算四个角点
    point1 = center - vector1 - vector2
    point2 = center + vector1 - vector2
    point3 = center + vector1 + vector2
    point4 = center - vector1 + vector2
    
    # 拼接所有点和分数
    polys = torch.cat([point1, point2, point3, point4], dim=-1)
    
    # 如果需要，可以添加 get_best_begin_point 的 PyTorch 实现
    polys = get_best_begin_point_tensor(polys)
    
    return polys

def get_best_begin_point_tensor(coordinates):
    """Get the best begin points of polygons using PyTorch tensors.

    Args:
        coordinates (Tensor): shape(n, 9).

    Returns:
        reorder coordinate (Tensor): shape(n, 9).
    """
    # 将张量转换为列表，处理每个多边形
    coordinates_list = []
    for i in range(coordinates.shape[0]):
        coord = coordinates[i]
        # 调用单样本处理函数
        processed_coord = get_best_begin_point_single_tensor(coord)
        coordinates_list.append(processed_coord)
    
    # 将处理后的列表堆叠回张量
    return torch.stack(coordinates_list, dim=0)






def poly2obb_tensor_oc_batch(polys):
    """Convert polygons to oriented bounding boxes using PyTorch tensors.

    Args:
        polys (Tensor): [n, 8] array of polygons

    Returns:
        obbs (Tensor): [n, 5] array of oriented bounding boxes
    """
    # 确保输入是张量
    if not isinstance(polys, torch.Tensor):
        polys = torch.tensor(polys, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 获取设备信息
    device = polys.device
    
    # 如果输入是单个多边形，转换为二维数组
    if polys.dim() == 1:
        polys = polys.unsqueeze(0)
    
    n = polys.shape[0]
    
    # 将数据移动到 CPU 进行处理（OpenCV 需要 CPU 数据）
    polys_cpu = polys.cpu().numpy()
    obbs_cpu = np.zeros((n, 5))
    
    for i in range(n):
        poly = polys_cpu[i]
        bboxps = poly.reshape((4, 2))
        rbbox = cv2.minAreaRect(bboxps)
        x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[2]
        
        if w < 2 or h < 2:
            # 对于无效的边界框，设置为特定值
            obbs_cpu[i] = [0, 0, 0, 0, 0]
            continue
            
        # 调整角度到 [0, 90] 范围
        while not 0 < a <= 90:
            if a == -90:
                a += 180
            else:
                a += 90
                w, h = h, w
                
        # 转换为弧度
        a = a / 180 * np.pi
        
        # 确保角度在正确范围内
        if not (0 < a <= np.pi / 2):
            a = a % (np.pi / 2)
            if a <= 0:
                a += np.pi / 2
                
        obbs_cpu[i] = [x, y, w, h, a]
    
    # 将结果移回原始设备
    obbs = torch.tensor(obbs_cpu, device=device, dtype=polys.dtype)
    return obbs



def get_best_begin_point_single_tensor(coordinate):
    """Get the best begin point of the single polygon using PyTorch tensors.

    Args:
        coordinate (Tensor): [x1, y1, x2, y2, x3, y3, x4, y4, score]

    Returns:
        reorder coordinate (Tensor): [x1, y1, x2, y2, x3, y3, x4, y4, score]
    """
    # 确保输入是张量
    if not isinstance(coordinate, torch.Tensor):
        coordinate = torch.tensor(coordinate, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 提取坐标和分数
    points = coordinate[:8].reshape(4, 2)
    score = coordinate[8:]
    
    # 计算边界框
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    xmin = torch.min(x_coords)
    ymin = torch.min(y_coords)
    xmax = torch.max(x_coords)
    ymax = torch.max(y_coords)
    
    # 创建四个不同的起点顺序组合
    combine = [
        points,
        torch.roll(points, shifts=-1, dims=0),
        torch.roll(points, shifts=-2, dims=0),
        torch.roll(points, shifts=-3, dims=0)
    ]
    
    # 目标矩形坐标
    dst_coordinate = torch.tensor([
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax]
    ], device=coordinate.device)
    
    # 计算每个组合与目标矩形的距离和
    force = torch.tensor(100000000.0, device=coordinate.device)
    force_flag = 0
    
    for i in range(4):
        temp_force = cal_line_length_tensor(combine[i][0], dst_coordinate[0]) + \
                     cal_line_length_tensor(combine[i][1], dst_coordinate[1]) + \
                     cal_line_length_tensor(combine[i][2], dst_coordinate[2]) + \
                     cal_line_length_tensor(combine[i][3], dst_coordinate[3])
        
        if temp_force < force:
            force = temp_force
            force_flag = i
    
    # 获取最佳组合
    best_combination = combine[force_flag]
    
    # 展平并添加分数
    result = torch.cat([best_combination.reshape(-1), score])
    
    return result






def cal_line_length_tensor(point1, point2):
    """Calculate the length of line using PyTorch tensors.

    Args:
        point1 (Tensor): [x,y] with shape (2,)
        point2 (Tensor): [x,y] with shape (2,)

    Returns:
        length (Tensor): Euclidean distance between the two points
    """
    # 确保输入是张量
    if not isinstance(point1, torch.Tensor):
        point1 = torch.tensor(point1, device='cuda' if torch.cuda.is_available() else 'cpu')
    if not isinstance(point2, torch.Tensor):
        point2 = torch.tensor(point2, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 确保两个点在同一设备上
    if point1.device != point2.device:
        point2 = point2.to(point1.device)
    
    # 计算欧氏距离
    return torch.sqrt(torch.sum((point1 - point2) ** 2))