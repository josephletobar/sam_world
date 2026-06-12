from sklearn.metrics.pairwise import cosine_similarity
import cv2
import numpy as np
from utils.create_object import WorldObject

MAX_ASSOCIATION_DISTANCE_M = 10
EMBEDDING_EMA_ALPHA = 0.8

class Association():

    def __init__(self, global_objects, graph_builder, threshold=0.95):

        self.global_objects = global_objects
        self.graph_builder = graph_builder

        self.threshold = threshold 
        self.observation_idx = 0
        self.current_timestamp = None

    def _label_prefix(self, obj: WorldObject):
        label_source = obj.node_id or obj.label
        return label_source.split("_", 1)[0]

    def _next_node_id(self, new_object: WorldObject):
        prefix = self._label_prefix(new_object)
        global_objects = self.global_objects or []
        count = sum(
            1
            for obj in global_objects
            if self._label_prefix(obj) == prefix
        )
        return f"{prefix}_{count}"

    def _fit_debug_image(self, image, width=360, height=360):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        if image is None or image.size == 0:
            return canvas

        if image.ndim == 2:
            image = self._depth_to_debug_image(image)

        image_h, image_w = image.shape[:2]
        scale = min(width / image_w, height / image_h)
        resized_w = max(1, int(round(image_w * scale)))
        resized_h = max(1, int(round(image_h * scale)))
        resized = cv2.resize(image, (resized_w, resized_h))
        x = (width - resized_w) // 2
        y = (height - resized_h) // 2
        canvas[y:y + resized_h, x:x + resized_w] = resized
        return canvas

    def _depth_to_debug_image(self, depth):
        valid_depth = depth[depth > 0]
        if len(valid_depth) == 0:
            return np.zeros((*depth.shape, 3), dtype=np.uint8)

        min_depth = valid_depth.min()
        max_depth = valid_depth.max()

        if max_depth == min_depth:
            normalized = np.zeros(depth.shape, dtype=np.uint8)
            normalized[depth > 0] = 255
        else:
            normalized = np.zeros(depth.shape, dtype=np.uint8)
            normalized[depth > 0] = np.clip(
                (depth[depth > 0] - min_depth) * 255 / (max_depth - min_depth),
                0,
                255
            ).astype(np.uint8)

        return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)

    def _show_debug_comparison(self, new_object, existing_object, title, score_text):
        new_panel = self._fit_debug_image(new_object.segmented_rgb)
        existing_panel = self._fit_debug_image(existing_object.segmented_rgb)
        new_depth_panel = self._fit_debug_image(new_object.segmented_depth)
        existing_depth_panel = self._fit_debug_image(existing_object.segmented_depth)

        rgb_row = np.hstack((new_panel, existing_panel))
        depth_row = np.hstack((new_depth_panel, existing_depth_panel))
        combined = np.vstack((rgb_row, depth_row))

        cv2.putText(
            combined,
            f"new: {new_object.label}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        cv2.putText(
            combined,
            f"existing: {existing_object.node_id}",
            (372, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        cv2.putText(
            combined,
            score_text,
            (12, combined.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
        cv2.putText(
            combined,
            "depth",
            (12, 388),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(title, combined)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 32:
                break
        cv2.destroyWindow(title)

    def _format_depth_stats(self, obj):
        stats = obj.depth_stats
        return (
            f"min={stats['min']:.3f}, "
            f"p25={stats['p25']:.3f}, "
            f"median={stats['median']:.3f}, "
            f"p75={stats['p75']:.3f}, "
            f"max={stats['max']:.3f}"
        )

    def _mark_new_object(self, obj):
        obj.first_seen = self.current_timestamp
        obj.last_seen = self.current_timestamp

    def _merge_object(self, existing_object, new_object):
        first_seen = existing_object.first_seen
        node_id = existing_object.node_id
        img_embedding = (
            EMBEDDING_EMA_ALPHA * existing_object.img_embedding +
            (1 - EMBEDDING_EMA_ALPHA) * new_object.img_embedding
        )
        txt_embedding = (
            EMBEDDING_EMA_ALPHA * existing_object.txt_embedding +
            (1 - EMBEDDING_EMA_ALPHA) * new_object.txt_embedding
        )

        existing_object.confidence = new_object.confidence
        existing_object.sam_mask = new_object.sam_mask
        existing_object.world_pos = new_object.world_pos
        existing_object.image_pos = new_object.image_pos
        existing_object.local_pos = new_object.local_pos
        existing_object.translation = new_object.translation
        existing_object.median_depth = new_object.median_depth
        existing_object.depth_stats = new_object.depth_stats
        existing_object.segmented_rgb = new_object.segmented_rgb
        existing_object.segmented_depth = new_object.segmented_depth
        existing_object.img_embedding = img_embedding
        existing_object.txt_embedding = txt_embedding
        existing_object.node_id = node_id
        existing_object.first_seen = (
            first_seen
            if first_seen is not None
            else self.current_timestamp
        )
        existing_object.last_seen = self.current_timestamp

        return existing_object
    
    def _associate(self, new_object: WorldObject, world_objects: list[WorldObject]):
        new_prefix = self._label_prefix(new_object)
        world_objects = [
            obj
            for obj in world_objects
            if self._label_prefix(obj) == new_prefix
        ]

        if len(world_objects) == 0:
            return True, self._next_node_id(new_object), None

        embedding = new_object.img_embedding
        segmented = new_object.segmented_rgb
        pose = new_object.world_pos

        embedding_matrix = [obj.img_embedding for obj in world_objects]
        segmented_objs = [obj.segmented_rgb for obj in world_objects]
        poses = [obj.world_pos for obj in world_objects]

        if len(embedding_matrix) == 0:
            return None, None

        CLIP_WEIGHT = 0.3
        POSE_WEIGHT = 0.1
        COLOR_WEIGHT = 0.3
        SHAPE_WEIGHT = 0.3

        # Color Score
        curr_color = segmented[segmented.any(axis=2)].mean(axis=0)
        old_colors = np.array([
            obj[obj.any(axis=2)].mean(axis=0)
            for obj in segmented_objs
        ])
        color_scores = np.exp(
            -np.linalg.norm(old_colors - curr_color, axis=1) / 100
        )

        # Geometric Shape Score
        curr_mask = cv2.resize(
            (segmented.any(axis=2)).astype(np.uint8),
            (64, 64)
        )
        old_masks = [
            cv2.resize(
                (obj.any(axis=2)).astype(np.uint8),
                (64, 64)
            )
            for obj in segmented_objs
        ]
        shape_scores = np.array([
            1 - cv2.absdiff(curr_mask, old_mask).mean()
            for old_mask in old_masks
        ])

        # Cosine Sims
        cosine_sims = cosine_similarity([embedding], embedding_matrix)[0]

        # Pose Score
        poses_np = np.array(poses)
        curr_pose = np.array(pose)
        dists = np.linalg.norm(
            poses_np - curr_pose,
            axis=1
        )

        # if np.min(dists) > MAX_ASSOCIATION_DISTANCE_M:


        #     closest_idx = np.argmin(dists)
        #     new_node_id = self._next_node_id(new_object)
        #     print(
        #         "\nASSOCIATION DISTANCE GATE\n"
        #         f"  candidate: {new_object.label} -> {world_objects[closest_idx].node_id}\n"
        #         f"  new_pose:           {new_object.world_pos}\n"
        #         f"  new_local_pos:      {new_object.local_pos}\n"
        #         f"  new_translation:    {new_object.translation}\n"
        #         f"  new_median_depth:   {new_object.median_depth:.3f}\n"
        #         f"  new_depth_stats:    {self._format_depth_stats(new_object)}\n"
        #         f"  existing_pose:      {world_objects[closest_idx].world_pos}\n"
        #         f"  existing_local_pos: {world_objects[closest_idx].local_pos}\n"
        #         f"  existing_translation: {world_objects[closest_idx].translation}\n"
        #         f"  existing_median_depth: {world_objects[closest_idx].median_depth:.3f}\n"
        #         f"  existing_depth_stats: {self._format_depth_stats(world_objects[closest_idx])}\n"
        #         f"  closest_distance_m: {dists[closest_idx]:.3f}\n"
        #         f"  max_distance_m:     {MAX_ASSOCIATION_DISTANCE_M:.3f}\n"
        #         f"  decision:           new object ({new_node_id})"
        #     )
        #     self._show_debug_comparison(
        #         new_object,
        #         world_objects[closest_idx],
        #         "Association New Object",
        #         f"distance {dists[closest_idx]:.2f}m > {MAX_ASSOCIATION_DISTANCE_M:.2f}m"
        #     )

        #     return True, new_node_id, None

        SIGMA = 2
        pose_scores = np.exp(-dists / SIGMA)

        # Final Probability
        probabilities = (
            CLIP_WEIGHT * cosine_sims +
            POSE_WEIGHT * pose_scores +
            COLOR_WEIGHT * color_scores +
            SHAPE_WEIGHT * shape_scores
        )

        best_idx = np.argmax(probabilities)

        best_prob = probabilities[best_idx]

        is_different = best_prob < self.threshold
        new_node_id = self._next_node_id(new_object)
        decision = (
            f"new object ({new_node_id})"
            if is_different
            else f"same as {world_objects[best_idx].node_id}"
        )

        print(
            "\nASSOCIATION SCORE\n"
            f"  candidate: {new_object.label} -> {world_objects[best_idx].node_id}\n"
            f"  new_pose:         {new_object.world_pos}\n"
            f"  new_local_pos:    {new_object.local_pos}\n"
            f"  new_translation:  {new_object.translation}\n"
            f"  new_median_depth: {new_object.median_depth:.3f}\n"
            f"  new_depth_stats:  {self._format_depth_stats(new_object)}\n"
            f"  old_pose:         {world_objects[best_idx].world_pos}\n"
            f"  old_local_pos:    {world_objects[best_idx].local_pos}\n"
            f"  old_translation:  {world_objects[best_idx].translation}\n"
            f"  old_median_depth: {world_objects[best_idx].median_depth:.3f}\n"
            f"  old_depth_stats:  {self._format_depth_stats(world_objects[best_idx])}\n"
            f"  clip:      {cosine_sims[best_idx]:.3f} * {CLIP_WEIGHT:.2f} = "
            f"{cosine_sims[best_idx] * CLIP_WEIGHT:.3f}\n"
            f"  pose:      {pose_scores[best_idx]:.3f} * {POSE_WEIGHT:.2f} = "
            f"{pose_scores[best_idx] * POSE_WEIGHT:.3f}\n"
            f"  color:     {color_scores[best_idx]:.3f} * {COLOR_WEIGHT:.2f} = "
            f"{color_scores[best_idx] * COLOR_WEIGHT:.3f}\n"
            f"  shape:     {shape_scores[best_idx]:.3f} * {SHAPE_WEIGHT:.2f} = "
            f"{shape_scores[best_idx] * SHAPE_WEIGHT:.3f}\n"
            f"  final:     {best_prob:.3f}\n"
            f"  threshold: {self.threshold:.3f}\n"
            f"  decision:  {decision}"
        )

        if is_different: # They are different

            self._show_debug_comparison(
                new_object,
                world_objects[best_idx],
                "Association New Object",
                f"score {best_prob:.3f} < threshold {self.threshold:.3f}"
            )

            return True, new_node_id, None
        else:
            return False, None, world_objects[best_idx]
        
    def update(self, new_object, timestamp=None):
        self.observation_idx += 1
        self.current_timestamp = timestamp

        if len(self.global_objects) > 0:
        
            different, new_object.node_id, matched_object = self._associate(new_object=new_object, world_objects=self.global_objects)

            try:    
                if different:
                    self._mark_new_object(new_object)
                    self.global_objects.append(new_object)
                    self.graph_builder.add_object(new_object, self.global_objects)

                elif not different:
                    self._merge_object(matched_object, new_object)
                    if self.graph_builder is not None:
                        self.graph_builder.update_object(matched_object)
            except Exception as e:
                print(e)

        else:
            try:
                new_object.node_id = self._next_node_id(new_object)
                self._mark_new_object(new_object)
                self.global_objects.append(new_object)
                self.graph_builder.add_object(new_object, self.global_objects)
            except Exception as e:
                print(e)
