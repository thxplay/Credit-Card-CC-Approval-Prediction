import streamlit as st

def aboutme():
    st.header("👨‍💼 Okaviantama Karunia Haris")

    col1, col2 = st.columns([2, 1])  # 2:1 ratio

    with col1:
        st.markdown(
            """
            <div style='text-align: justify;'>
            Saya adalah lulusan <b>Teknik Informatika</b> yang memiliki minat besar di bidang <b>Data Science</b>. 
            Memiliki pengalaman kerja di bidang Administrasi, Komunikasi, dan Jaringan, yang erat kaitannya dengan industri IT, 
            saya terbiasa bekerja secara analitis dan terstruktur.
            <br><br>
            Saya terus mengembangkan kemampuan melalui berbagai bootcamp intensif dan sertifikasi, 
            sebagai bentuk komitmen terhadap pembelajaran berkelanjutan.
            <br><br>
            Tujuan karier saya adalah menjadi <b>Data Scientist</b> yang mampu menghadirkan insight berbasis data 
            dan mendukung pengambilan keputusan strategis yang berdampak langsung pada pertumbuhan bisnis.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.subheader("📬 Kontak & Profil")
        st.markdown("- 📧 Email: okaviantama.haris@gmail.com")
        st.markdown("- 💼 [LinkedIn](https://www.linkedin.com/in/okaviantama-karunia-haris/)")
        st.markdown("- 🌐 [Website](https://thxplay.blogspot.com/)")
        st.markdown("- 🐙 [GitHub](https://github.com/thxplay)")

    with col2:
        st.image("images/myprofil.png", width=500, caption="Okaviantama Karunia Haris")  # Ganti dengan foto profil atau logo