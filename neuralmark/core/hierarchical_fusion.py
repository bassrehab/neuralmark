import tensorflow as tf
from tensorflow.keras import layers, Model


class LowLevelFusion(layers.Layer):
    """Handles fusion of low-level features (edges, textures, colors)."""

    def __init__(self, feature_dim=64):
        super(LowLevelFusion, self).__init__()

        # Edge attention
        self.edge_conv = layers.Conv2D(32, 3, padding='same')
        self.edge_attention = layers.Dense(1, activation='sigmoid')

        # Texture attention
        self.texture_conv = layers.Conv2D(32, 3, padding='same')
        self.texture_attention = layers.Dense(1, activation='sigmoid')

        # Color attention
        self.color_conv = layers.Conv2D(32, 1, padding='same')
        self.color_attention = layers.Dense(1, activation='sigmoid')

        # Feature fusion
        self.fusion_conv = layers.Conv2D(feature_dim, 1, padding='same')
        self.layer_norm = layers.LayerNormalization()

    def call(self, inputs):
        # Edge processing
        edge_features = self.edge_conv(inputs)
        edge_weights = self.edge_attention(tf.reshape(edge_features, [tf.shape(edge_features)[0], -1]))
        edge_attended = edge_features * tf.reshape(edge_weights, [-1, 1, 1, 1])

        # Texture processing
        texture_features = self.texture_conv(inputs)
        texture_weights = self.texture_attention(tf.reshape(texture_features, [tf.shape(texture_features)[0], -1]))
        texture_attended = texture_features * tf.reshape(texture_weights, [-1, 1, 1, 1])

        # Color processing
        color_features = self.color_conv(inputs)
        color_weights = self.color_attention(tf.reshape(color_features, [tf.shape(color_features)[0], -1]))
        color_attended = color_features * tf.reshape(color_weights, [-1, 1, 1, 1])

        # Combine features
        combined = tf.concat([edge_attended, texture_attended, color_attended], axis=-1)
        fused = self.fusion_conv(combined)

        return self.layer_norm(fused)


class MidLevelFusion(layers.Layer):
    """Handles fusion of mid-level features (structures, patterns, regions)."""

    def __init__(self, feature_dim=128):
        super(MidLevelFusion, self).__init__()

        # Structure attention
        self.structure_conv = layers.Conv2D(64, 3, padding='same')
        self.structure_pool = layers.MaxPooling2D(2)
        self.structure_attention = layers.Dense(1, activation='sigmoid')

        # Pattern attention
        self.pattern_conv = layers.Conv2D(64, 3, padding='same')
        self.pattern_attention = layers.Dense(1, activation='sigmoid')

        # Region attention
        self.region_conv = layers.Conv2D(64, 3, padding='same')
        self.region_attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)

        # Feature fusion
        self.fusion_dense = layers.Dense(feature_dim)
        self.layer_norm = layers.LayerNormalization()

    def call(self, inputs, low_level_features):
        # Structure processing
        structure_features = self.structure_conv(inputs)
        structure_features = self.structure_pool(structure_features)
        structure_weights = self.structure_attention(
            tf.reshape(structure_features, [tf.shape(structure_features)[0], -1])
        )
        structure_attended = structure_features * tf.reshape(structure_weights, [-1, 1, 1, 1])

        # Pattern processing
        pattern_features = self.pattern_conv(inputs)
        pattern_weights = self.pattern_attention(
            tf.reshape(pattern_features, [tf.shape(pattern_features)[0], -1])
        )
        pattern_attended = pattern_features * tf.reshape(pattern_weights, [-1, 1, 1, 1])

        # Region processing with attention
        region_features = self.region_conv(inputs)
        region_features = tf.reshape(region_features, [tf.shape(region_features)[0], -1, 64])
        region_attended = self.region_attention(region_features, region_features, region_features)

        # Combine with low-level features
        combined = tf.concat([
            tf.reshape(structure_attended, [tf.shape(inputs)[0], -1]),
            tf.reshape(pattern_attended, [tf.shape(inputs)[0], -1]),
            tf.reshape(region_attended, [tf.shape(inputs)[0], -1]),
            tf.reshape(low_level_features, [tf.shape(inputs)[0], -1])
        ], axis=-1)

        # Fuse features
        fused = self.fusion_dense(combined)
        return self.layer_norm(fused)


class HighLevelFusion(layers.Layer):
    """Handles fusion of high-level features (semantics, content, context)."""

    def __init__(self, feature_dim=256):
        super(HighLevelFusion, self).__init__()

        # Semantic attention
        self.semantic_conv = layers.Conv2D(128, 3, padding='same')
        self.semantic_attention = layers.MultiHeadAttention(num_heads=8, key_dim=64)

        # Content attention
        self.content_dense = layers.Dense(128)
        self.content_attention = layers.Dense(1, activation='sigmoid')

        # Context attention
        self.context_conv = layers.Conv2D(128, 3, padding='same', dilation_rate=2)
        self.context_attention = layers.Dense(1, activation='sigmoid')

        # Feature fusion
        self.fusion_dense = layers.Dense(feature_dim)
        self.layer_norm = layers.LayerNormalization()

    def call(self, inputs, mid_level_features):
        # Semantic processing
        semantic_features = self.semantic_conv(inputs)
        semantic_features = tf.reshape(semantic_features, [tf.shape(semantic_features)[0], -1, 128])
        semantic_attended = self.semantic_attention(semantic_features, semantic_features, semantic_features)

        # Content processing
        content_features = self.content_dense(tf.reshape(inputs, [tf.shape(inputs)[0], -1, tf.shape(inputs)[-1]]))
        content_weights = self.content_attention(content_features)
        content_attended = content_features * content_weights

        # Context processing
        context_features = self.context_conv(inputs)
        context_weights = self.context_attention(tf.reshape(context_features, [tf.shape(context_features)[0], -1]))
        context_attended = context_features * tf.reshape(context_weights, [-1, 1, 1, 1])

        # Combine with mid-level features
        combined = tf.concat([
            tf.reshape(semantic_attended, [tf.shape(inputs)[0], -1]),
            tf.reshape(content_attended, [tf.shape(inputs)[0], -1]),
            tf.reshape(context_attended, [tf.shape(inputs)[0], -1]),
            mid_level_features
        ], axis=-1)

        # Fuse features
        fused = self.fusion_dense(combined)
        return self.layer_norm(fused)


class HierarchicalFusion(Model):
    """Main hierarchical fusion model combining all levels of feature fusion."""

    def __init__(self, low_dim=64, mid_dim=128, high_dim=256):
        super(HierarchicalFusion, self).__init__()

        # Initialize fusion layers
        self.low_level = LowLevelFusion(low_dim)
        self.mid_level = MidLevelFusion(mid_dim)
        self.high_level = HighLevelFusion(high_dim)

        # Feature refinement
        self.refinement_conv = layers.Conv2D(high_dim, 1, padding='same')
        self.output_norm = layers.LayerNormalization()

    def call(self, inputs, domain_features):
        # Low-level fusion
        low_level_features = self.low_level(inputs)

        # Mid-level fusion with low-level features
        mid_level_features = self.mid_level(inputs, low_level_features)

        # High-level fusion with mid-level features
        high_level_features = self.high_level(inputs, mid_level_features)

        # Combine with domain features
        combined = tf.concat([
            tf.reshape(high_level_features, [tf.shape(inputs)[0], -1]),
            tf.reshape(domain_features, [tf.shape(inputs)[0], -1])
        ], axis=-1)

        # Final refinement
        refined = self.refinement_conv(
            tf.reshape(combined, [tf.shape(inputs)[0], 1, 1, -1])
        )

        return self.output_norm(refined), {
            'low_level': low_level_features,
            'mid_level': mid_level_features,
            'high_level': high_level_features
        }