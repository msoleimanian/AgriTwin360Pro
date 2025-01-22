import streamlit as st
from streamlit_option_menu import option_menu
import Performance_Traints as performance_traits

import Sensors as sen

def constructoemain():
    selected = option_menu(None, ["Traits", "Sensors"],
        menu_icon="cast", default_index=0, orientation="horizontal")

    if selected == "Traits":
        performance_traits.insightConstructor()

    if selected == "Sensors":
        sen.exploreConstructor()
