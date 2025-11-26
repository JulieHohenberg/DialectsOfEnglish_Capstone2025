import streamlit as st
import pandas as pd

st.set_page_config(page_title="Predictions", layout="wide")
st.markdown("<h1 style='text-align: center;'>American Dialect Prediction Quiz</h1>", unsafe_allow_html=True)
st.write("""<p style='text-align: center; font-size: 1.3rem;'>Take this quiz to discover which American dialect region you're from based on your word choices!
Answer 10 questions about everyday terms and we'll predict your regional dialect.""", unsafe_allow_html=True)

QUESTIONS = {
    303: {
        "text": 'What do you call the drink made with milk + ice cream?',
        "choices": [
            "milkshake/shake", "frappe", "cabinet", "velvet", "thick shake", "other"
        ]
    },
    300: {
        "text": 'Grass between sidewalk + road?',
        "choices": [
            "berm", "parking", "tree lawn", "terrace", "curb strip", "beltway", "verge", "other"
        ]
    },
    335: {
        "text": 'What is “the City”?',
        "choices": ["New York City", "Boston", "DC", "LA", "Chicago", "other"]
    },
    358: {
        "text": 'Drive-through liquor store?',
        "choices": [
            "party barn", "brew thru", "bootlegger", "beer barn", "beverage barn",
            "no special term", "never heard", "other"
        ]
    },
    316: {
        "text": 'Diagonal across the street?',
        "choices": [
            "kitty-corner", "kitacorner", "catercorner", "catty-corner",
            "kitty cross", "kitty wampus", "diagonal", "other"
        ]
    },
    350: {
        "text": 'Night before Halloween?',
        "choices": [
            "mischief night", "devil's night", "cabbage night", "goosy night",
            "gate night", "trick night", "I have no word", "other"
        ]
    },
    343: {
        "text": 'Thing you drink water from in school?',
        "choices": [
            "bubbler", "drinking fountain", "water fountain", "water bubbler", "other"
        ]
    },
    319: {
        "text": 'General term for a big road you drive fast on?',
        "choices": [
            "highway", "freeway", "parkway", "turnpike", "expressway",
            "throughway/thru-way", "other"
        ]
    },
    302: {
        "text": 'Median of a divided highway?',
        "choices": [
            "median", "median strip", "neutral ground", "mall",
            "traffic island", "island", "park strip", "other"
        ]
    },
    305: {
        "text": 'Glow-in-the-dark bug?',
        "choices": [
            "lightning bug", "firefly", "both", "peenie wallie",
            "I have no word", "other"
        ]
    }
}

REGION_INFO = {
    "The West": "Your dialect aligns with the Western United States, including California, Oregon, Washington, and the Mountain states.",
    "North Central": "Your dialect matches the North Central region, including Wisconsin, Minnesota, and parts of the Upper Midwest.",
    "Northern New England": "Your speech patterns align with Northern New England, including Vermont, New Hampshire, and Maine.",
    "The North": "Your dialect is characteristic of the Northern United States, including Pennsylvania, Ohio, and parts of the Great Lakes region.",
    "Greater New York City": "Your dialect matches the Greater New York City area, including New York City and surrounding areas.",
    "Midland": "Your dialect aligns with the Midland region, a transitional area between North and South.",
    "The South": "Your speech patterns match the Southern United States dialect region."
}

if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'quiz_complete' not in st.session_state:
    st.session_state.quiz_complete = False

# Function to convert answers to one-hot encoded vector
def answers_to_vector(answers, all_features):
    one_hot = {f"{qid}_{ans}": 1 for qid, ans in answers.items()}
    
    row = pd.DataFrame([one_hot])
    row_aligned = row.reindex(columns=all_features, fill_value=0)
    
    return row_aligned

# Prediction function matching notebook approach
def predict_region(answers):
    """
    Predict region based on answers using model approach from Colab notebook
    
    In production with saved models:
    1. Load model: model = joblib.load('logreg_region.pkl') or joblib.load('xgb_model.pkl')
    2. Load encoder: encoder = joblib.load('region_label_encoder.pkl')
    3. Load features: feature_cols = joblib.load('feature_columns.pkl')
    4. Then run:
        X_user = answers_to_vector(answers, feature_cols)
        pred_class = model.predict(X_user)[0]
        region = encoder.inverse_transform([pred_class])[0]
        proba = model.predict_proba(X_user)[0]
    """
    
    all_features = []
    for qid, q_data in QUESTIONS.items():
        for choice in q_data['choices']:
            all_features.append(f"{qid}_{choice}")
    
    X_user = answers_to_vector(answers, all_features)
    
    # Simplified prediction using key features (demo version)
    region_scores = {
        "The West": 0,
        "North Central": 0,
        "Northern New England": 0,
        "The North": 0,
        "Greater New York City": 0,
        "Midland": 0,
        "The South": 0
    }
    
    # Apply feature weights (simplified from actual model coefficients)
    feature_weights = {
        "303_frappe": {"Northern New England": 3.5},
        "303_cabinet": {"Northern New England": 2.8},
        "300_parking": {"The West": 2.2},
        "300_tree lawn": {"The North": 2.1},
        "300_terrace": {"North Central": 2.0},
        "335_New York City": {"Greater New York City": 3.8, "The North": 0.8},
        "335_LA": {"The West": 3.5},
        "335_Chicago": {"North Central": 3.3},
        "335_Boston": {"Northern New England": 3.4},
        "343_bubbler": {"North Central": 3.2, "Northern New England": 2.3},
        "343_water fountain": {"The South": 1.2, "Greater New York City": 0.9},
        "305_lightning bug": {"The South": 2.1, "Midland": 1.3, "North Central": 0.8},
        "305_firefly": {"The West": 1.1, "Northern New England": 0.9},
        "319_freeway": {"The West": 2.3},
        "319_highway": {"Midland": 1.1, "The South": 0.9},
        "350_devil's night": {"North Central": 2.2},
        "350_mischief night": {"The North": 2.0, "Greater New York City": 1.2},
        "316_kitty-corner": {"North Central": 1.2, "The West": 0.9},
        "316_catercorner": {"North Central": 1.1, "The West": 0.9},
        "302_neutral ground": {"The South": 2.5},
    }
    
    for feature_name in X_user.columns[X_user.iloc[0] == 1]:
        if feature_name in feature_weights:
            for region, weight in feature_weights[feature_name].items():
                region_scores[region] += weight
    
    for region in region_scores:
        region_scores[region] += 0.5
    
    predicted_region = max(region_scores, key=region_scores.get)
    
    # Calculate confidence (similar to predict_proba)
    total_score = sum(region_scores.values())
    confidence = (region_scores[predicted_region] / total_score) * 100
    
    return predicted_region, confidence, region_scores

st.markdown("---")
st.subheader("Questions")

question_ids = list(QUESTIONS.keys())

for i, qid in enumerate(question_ids):
    question = QUESTIONS[qid]
    st.markdown(f"**Question {i+1} of {len(question_ids)}**")
    st.markdown(f"*{question['text']}*")
    
    answer = st.radio(
        "Select your answer:",
        question['choices'],
        key=f"q_{qid}",
        index=None
    )
    
    if answer:
        st.session_state.answers[qid] = answer
    
    st.markdown("---")

all_answered = len(st.session_state.answers) == len(QUESTIONS)

if all_answered:
    if st.button("🎯 Get My Results!", type="primary", use_container_width=True):
        st.session_state.quiz_complete = True

if st.session_state.quiz_complete and all_answered:
    st.markdown("---")
    st.header("Your Results")
    
    predicted_region, confidence, all_scores = predict_region(st.session_state.answers)
    
    # Display prediction
    st.success(f"### Your predicted dialect region: **{predicted_region}**")
    st.markdown(REGION_INFO.get(predicted_region, ""))
    
    st.metric("Confidence", f"{confidence:.1f}%")
    
    with st.expander("See detailed breakdown"):
        st.markdown("**Regional match scores:**")
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        for region, score in sorted_scores:
            normalized_score = (score / sum(all_scores.values())) * 100
            st.progress(normalized_score / 100, text=f"{region}: {normalized_score:.1f}%")
    
    if st.button("Take Quiz Again"):
        st.session_state.answers = {}
        st.session_state.quiz_complete = False
        st.rerun()

elif not all_answered:
    questions_remaining = len(QUESTIONS) - len(st.session_state.answers)
    st.info(f"Please answer all questions ({questions_remaining} remaining) to see your results.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
<p>Results are predictions based on common speech patterns and may not reflect individual variation.</p>
</div>
""", unsafe_allow_html=True)