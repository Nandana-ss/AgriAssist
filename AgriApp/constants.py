
growth_stage_durations = {
    'Rice_seed_level': 7,  # Days to move to the next stage
    'Rice_Germination_stage': 14,
    'Rice_Seedling_Stage': 21,
    'Rice_Grain_Filling_Stage': 28,
    'Rice_Dough_Stage': 35,
    'Rice_Harvesting_stage': 0,  # End stage
}

growth_stages = {
    'Rice_seed_level': 'Rice_Germination_stage',
    'Rice_Germination_stage': 'Rice_Seedling_Stage',
    'Rice_Seedling_Stage': 'Rice_Grain_Filling_Stage',
    'Rice_Grain_Filling_Stage': 'Rice_Dough_Stage',
    'Rice_Dough_Stage': 'Rice_Harvesting_stage',
}

care_instructions = {
    'Rice_seed_level': {
        'Watering': 'Keep the soil moist but not waterlogged.',
        'Fertilizer': 'Use a balanced starter fertilizer.',
        'Pesticide': 'Monitor for pests and apply appropriate insecticide if necessary.',
    },
    'Rice_Germination_stage': {
        'Watering': 'Maintain constant moisture in the soil.',
        'Fertilizer': 'Apply a nitrogen-rich fertilizer to support early growth.',
        'Pesticide': 'Watch for fungal infections and treat as needed.',
    },
    'Rice_Seedling_Stage': {
        'Watering': 'Ensure the soil is consistently moist.',
        'Fertilizer': 'Use a balanced fertilizer with more nitrogen.',
        'Pesticide': 'Check for pests like aphids or leafhoppers.',
    },
    'Rice_Grain_Filling_Stage': {
        'Watering': 'Reduce watering as the grains start filling.',
        'Fertilizer': 'Apply a low-nitrogen fertilizer to avoid excessive growth.',
        'Pesticide': 'Monitor for diseases like leaf blast or bacterial blight.',
    },
    'Rice_Dough_Stage': {
        'Watering': 'Withhold water to allow grains to harden.',
        'Fertilizer': 'No additional fertilization needed.',
        'Pesticide': 'Ensure no fungal or pest issues are present.',
    },
    'Rice_Harvesting_stage': {
        'Watering': 'No watering needed.',
        'Fertilizer': 'No additional fertilization needed.',
        'Pesticide': 'Harvest timely to avoid pest infestations.',
    },
}
