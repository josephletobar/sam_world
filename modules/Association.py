from sklearn.metrics.pairwise import cosine_similarity
import cv2
import numpy as np
from modules.ObjectPerception import WorldObject


class Association():

    def __init__(self, global_objects, graph_builder, threshold=0.95):

        self.global_objects = global_objects
        self.graph_builder = graph_builder

        self.threshold = threshold 

    
    def _associate(self, new_object: WorldObject, world_objects: list[WorldObject]):
        embedding = new_object.img_embedding
        segmented = new_object.segmented_rgb
        pose = new_object.world_pos

        embedding_matrix = [obj.img_embedding for obj in world_objects]
        segmented_objs = [obj.segmented_rgb for obj in world_objects]
        poses = [obj.world_pos for obj in world_objects]

        if len(embedding_matrix) == 0:
            return None, None

        CLIP_WEIGHT = 0.6
        POSE_WEIGHT = 0.3
        COLOR_WEIGHT = 0.05
        SHAPE_WEIGHT = 0.05

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

        if best_prob < self.threshold:
            node_id = f"{new_object.label}_{sum(1 for obj in world_objects if obj.label == new_object.label)}"

            return True, node_id
        else:
            return False, None
        
    def update(self, new_object):

        if len(self.global_objects) > 0:
        
            different, new_object.node_id = self._associate(new_object=new_object, world_objects=self.global_objects)

            try:    
                if different:
                    self.global_objects.append(new_object)
                    self.graph_builder.add_object(new_object, self.global_objects)

                elif not different:
                    # Merge Nodes (skip for now)
                    pass
            except Exception as e:
                print(e)

        else:
            try:
                new_object.node_id = f"{new_object.label}_{sum(1 for o in self.global_objects if o.label == new_object.label)}"
                self.global_objects.append(new_object)
                self.graph_builder.add_object(new_object, self.global_objects)
            except Exception as e:
                print(e)