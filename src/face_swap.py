import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh


def get_face_landmarks(image, face_mesh):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0].landmark
    h, w = image.shape[:2]
    pts = np.array([(int(l.x * w), int(l.y * h)) for l in landmarks])
    return pts


def apply_affine(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None,
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst


def warp_triangle(src, dst, t_src, t_dst):
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))

    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(3):
        t1_rect.append(((t_src[i][0] - r1[0]), (t_src[i][1] - r1[1])))
        t2_rect.append(((t_dst[i][0] - r2[0]), (t_dst[i][1] - r2[1])))
        t2_rect_int.append(((t_dst[i][0] - r2[0]), (t_dst[i][1] - r2[1])))

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    img1_rect = src[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    size = (r2[2], r2[3])
    img2_rect = apply_affine(img1_rect, t1_rect, t2_rect, size)

    dst_rect = dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    dst_rect = dst_rect * (1 - mask) + img2_rect * mask
    dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = dst_rect


def delaunay_triangulation(points, shape):
    subdiv = cv2.Subdiv2D((0, 0, shape[1], shape[0]))
    for p in points:
        subdiv.insert((int(p[0]), int(p[1])))
    triangle_list = subdiv.getTriangleList()
    delaunay_tri = []
    for t in triangle_list:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        idx = []
        for p in pts:
            for i, point in enumerate(points):
                if abs(p[0] - point[0]) < 1 and abs(p[1] - point[1]) < 1:
                    idx.append(i)
        if len(idx) == 3:
            delaunay_tri.append(tuple(idx))
    return delaunay_tri


def face_swap(source_img, target_img):
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        src_pts = get_face_landmarks(source_img, face_mesh)
        dst_pts = get_face_landmarks(target_img, face_mesh)

    if src_pts is None or dst_pts is None:
        raise ValueError("Could not detect face landmarks in one of the images")

    target_output = target_img.copy()
    triangles = delaunay_triangulation(dst_pts, target_img.shape)

    for tri in triangles:
        t_src = [src_pts[i] for i in tri]
        t_dst = [dst_pts[i] for i in tri]
        warp_triangle(source_img, target_output, t_src, t_dst)

    hull = cv2.convexHull(np.float32(dst_pts))
    mask = np.zeros(target_img.shape, dtype=target_img.dtype)
    cv2.fillConvexPoly(mask, np.int32(hull), (255, 255, 255))

    r = cv2.boundingRect(hull)
    center = (int((r[0] + r[0] + r[2]) / 2), int((r[1] + r[1] + r[3]) / 2))
    output = cv2.seamlessClone(target_output, target_img, mask, center, cv2.NORMAL_CLONE)
    return output


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Face Swap using MediaPipe")
    parser.add_argument('source', help='Path to source face image')
    parser.add_argument('target', help='Path to target image')
    parser.add_argument('--output', default='output.jpg', help='Path for saving result')
    args = parser.parse_args()

    src = cv2.imread(args.source)
    dst = cv2.imread(args.target)
    if src is None or dst is None:
        raise ValueError('Could not load input images')

    result = face_swap(src, dst)
    cv2.imwrite(args.output, result)
    print(f'Saved output to {args.output}')


if __name__ == '__main__':
    main()
