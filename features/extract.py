"""Video-only features for one watched zone: density, velocity entropy, slowdown.

Density here is a classical proxy (foreground area / person area) so the pipeline
runs offline. Swap `density()` for a pretrained counter (e.g. DM-Count) on real video.
"""
import cv2
import numpy as np

PERSON_AREA_M2 = np.pi * 0.2 ** 2


class FeatureExtractor:
    def __init__(self, zone_px, px_per_m, fps, n_bins=8, v_free=1.2, fg_thresh=60):
        self.x0, self.y0, self.x1, self.y1 = zone_px
        self.px_per_m, self.fps = px_per_m, fps
        self.n_bins, self.v_free, self.fg_thresh = n_bins, v_free, fg_thresh
        self.zone_area_m2 = (self.x1 - self.x0) * (self.y1 - self.y0) / px_per_m ** 2
        self.prev = None

    def density(self, roi):
        """People per m^2 from the share of foreground pixels."""
        fg = roi > self.fg_thresh
        return fg.mean() * self.zone_area_m2 / PERSON_AREA_M2 / self.zone_area_m2, fg

    def update(self, frame):
        roi = frame[self.y0:self.y1, self.x0:self.x1]
        dens, fg = self.density(roi)
        out = {"density": dens, "entropy": 0.0, "slowdown": 0.0, "speed": np.nan}
        if self.prev is not None:
            flow = cv2.calcOpticalFlowFarneback(self.prev, roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            if fg.mean() > 0.02:
                speed = float(mag[fg].mean()) * self.fps / self.px_per_m      # m/s
                out["speed"] = speed
                out["slowdown"] = 1.0 - min(speed / self.v_free, 1.0)
                w = mag[fg]
                if w.sum() > 1e-6:
                    hist, _ = np.histogram(ang[fg], bins=self.n_bins,
                                           range=(0, 2 * np.pi), weights=w)
                    p = hist / hist.sum()
                    nz = p[p > 0]
                    out["entropy"] = float(-(nz * np.log2(nz)).sum() / np.log2(self.n_bins))
        self.prev = roi.copy()
        return out
