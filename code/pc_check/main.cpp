/* main.cpp - PC 端验证固件检测器的测试壳
 * 用法: test.exe <frame.bin> <w> <h>
 * 读入原始灰度帧, 调用固件的 TagDetector_Detect, 打印 JSON 风格结果。
 */
#include "tag_detector.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: test.exe frame.bin w h\n"); return 2; }
    int w = atoi(argv[2]), h = atoi(argv[3]);
    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "open failed\n"); return 2; }
    uint8_t *buf = (uint8_t *)malloc((size_t)w * h);
    if (fread(buf, 1, (size_t)w * h, f) != (size_t)w * h) { fclose(f); return 2; }
    fclose(f);

    TagObservation o;
    bool ok = TagDetector_Detect(buf, w, h, w, &o);
    free(buf);
    if (!ok) { printf("FAIL no_tag\n"); return 1; }

    printf("OK id=%d hamming=%d u=%.2f v=%.2f theta_deg=%.3f edge_a=%.2f edge_p=%.2f "
           "margin=%.1f contrast=%.1f corners=[",
           o.id, o.hamming, o.u, o.v, o.theta * 180.0 / M_PI, o.edge_a, o.edge_p,
           o.margin, o.contrast);
    for (int i = 0; i < 4; i++)
        printf("(%.2f,%.2f)%s", o.corners[i][0], o.corners[i][1], i < 3 ? "," : "");
    printf("]\n");
    return 0;
}
