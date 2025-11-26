import streamlit as st

st.set_page_config(page_title="Dialect App", layout="wide")

def landing_page():

    st.set_page_config(layout="wide")

    # Title Section
    st.markdown("""
    <h1 style='text-align: center; font-size: 3rem;'>English Dialect App</h1>
    <p style='text-align: center; font-size: 1.3rem;'>
        Explore our detailed datasets of dialect variation in the English-speaking world.
    </p>
    """, unsafe_allow_html=True)

  
    st.markdown("---")

    # Data Source Section
    st.subheader("Data Source")
    st.markdown("""
    <div style="background-color: #F8F9FA; padding: 20px; border-radius: 12px;">
        <b>Dr. Bert Vaux</b>, a linguistics professor from the University of Cambridge, collected English dialect data for decades.<br><br>
        • Early surveys were done on <b>paper</b> and manually transcribed<br>
        • Later surveys were collected <b>digitally</b> by CS students<br>
        • Questions include both <b>text prompts</b> and <b>images</b><br><br>
        This dataset reflects natural, real-world variation, but also comes with noise and duplicates.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # What’s in the Dataset Section
    st.subheader("What’s in the Dataset?")
    st.write("Here’s an overview of the four main components of the survey:")

    colA, colB = st.columns(2)
    colC, colD = st.columns(2)

    with colA:
        st.markdown("""
        <div style="background-color:#FFE5E5; padding:20px; border-radius:15px;">
            <h3>Users</h3>
            <ul>
                <li>User ID + basic info</li>
                <li>Location + email</li>
                <li>Language background</li>
                <li><b>~360,000 users</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown("""
        <div style="background-color:#E2F7E1; padding:20px; border-radius:15px;">
            <h3>Questions</h3>
            <ul>
                <li><b>165 dialect questions</b></li>
                <li>Some include images</li>
                <li>6K–285M responses/question</li>
                <li>~65K average per question</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with colC:
        st.markdown("""
        <div style="background-color:#FFF3C4; padding:20px; border-radius:15px;">
            <h3>Choices</h3>
            <ul>
                <li>1–59 choices per question</li>
                <li>~8 choices on average</li>
                <li>Example: hero, hoagie, grinder...</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with colD:
        st.markdown("""
        <div style="background-color:#DDEBFF; padding:20px; border-radius:15px;">
            <h3>Responses</h3>
            <ul>
                <li>Links user → question → choice</li>
                <li>Can include free-text</li>
                <li>Massive variability per question</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    landing_page()

