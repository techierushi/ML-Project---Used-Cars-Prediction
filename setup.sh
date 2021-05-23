mkdir -p ~/.streamlit/

echo "[theme] 
primaryColor = '#919e8b'  
backgroundColor = rgba(254,248,239,1)'  
secondaryBackgroundColor =  '#ebd2b9'  
textColor = '#6e7074'
font = 'sans serif'
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml

