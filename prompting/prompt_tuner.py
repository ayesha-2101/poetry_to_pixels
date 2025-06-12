def tune_prompt(emotion, visuals, theme):
    visuals_joined = ", ".join(visuals[:3])
    return (
        f"Formulate a prompt to generate an image that reflects the poem’s emotional depth "
        f"and themes using the visual elements: {visuals_joined}. "
        f"Emotion: {emotion}. Theme: {theme}. Keep the prompt under 50 words."
    )
