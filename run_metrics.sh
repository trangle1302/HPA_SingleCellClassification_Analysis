INPUT_ANNOTATIONS_BOXES=/path/to/challenge-2018-train-vrd.csv
INPUT_ANNOTATIONS_LABELS=/path/to/challenge-2018-train-vrd-labels.csv
INPUT_PREDICTIONS=/path/to/predictions.csv
INPUT_CLASS_LABELMAP=/path/to/oid_object_detection_challenge_500_label_map.pbtxt
INPUT_RELATIONSHIP_LABELMAP=/path/to/relationships_labelmap.pbtxt
OUTPUT_METRICS=/path/to/output/metrics/file

echo "item { name: '/m/02gy9n' id: 602 display_name: 'Transparent' }
item { name: '/m/05z87' id: 603 display_name: 'Plastic' }
item { name: '/m/0dnr7' id: 604 display_name: '(made of)Textile' }
item { name: '/m/04lbp' id: 605 display_name: '(made of)Leather' }
item { name: '/m/083vt' id: 606 display_name: 'Wooden'}
">>${INPUT_CLASS_LABELMAP}

echo "item { name: 'at' id: 1 display_name: 'at' }
item { name: 'on' id: 2 display_name: 'on (top of)' }
item { name: 'holds' id: 3 display_name: 'holds' }
item { name: 'plays' id: 4 display_name: 'plays' }
item { name: 'interacts_with' id: 5 display_name: 'interacts with' }
item { name: 'wears' id: 6 display_name: 'wears' }
item { name: 'is' id: 7 display_name: 'is' }
item { name: 'inside_of' id: 8 display_name: 'inside of' }
item { name: 'under' id: 9 display_name: 'under' }
item { name: 'hits' id: 10 display_name: 'hits' }
"> ${INPUT_RELATIONSHIP_LABELMAP}

python object_detection/metrics/oid_vrd_challenge_evaluation.py \
    --input_annotations_boxes=${INPUT_ANNOTATIONS_BOXES} \
    --input_annotations_labels=${INPUT_ANNOTATIONS_LABELS} \
    --input_predictions=${INPUT_PREDICTIONS} \
    --input_class_labelmap=${INPUT_CLASS_LABELMAP} \
    --input_relationship_labelmap=${INPUT_RELATIONSHIP_LABELMAP} \
    --output_metrics=${OUTPUT_METRICS}