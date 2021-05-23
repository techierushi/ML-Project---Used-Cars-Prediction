import pandas as pd
import numpy as np
import xgboost
import streamlit as st

html_temp = """
<div style ="background-color:yellow;padding:13px">
<h1 style ="color:black;text-align:center;"> Used Car Price Prediction Portal </h1>
</div>
"""
st.markdown(html_temp, unsafe_allow_html = True)

st.markdown("""
<h3><i><center> We provide you with the best estimated price for your car </center></i></h3>
"""
, unsafe_allow_html = True)

st.header("Please fill in the required details appropriately")

#Prediction function
def prediction(df):

    preds = model.predict(df)
    return preds

# this is the main function in which we define our webpage
def user_input_features():

    Make = st.selectbox(
           'Select the car brand',
            ('Acura', 'Audi', 'BMW', 'Buick', 'Cadillac', 'Chevrolet',
            'Chrysler', 'Dodge', 'Ford', 'GMC', 'Honda', 'Hummer', 'Hyundai',
            'Infiniti', 'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Land Rover',
            'Lexus', 'Lincoln', 'MINI', 'Mazda', 'Mercedes-Benz', 'Mercury',
            'Mitsubishi', 'Nissan', 'Oldsmobile', 'Pontiac', 'Porsche', 'Saab',
            'Saturn', 'Scion', 'Subaru', 'Suzuki', 'Toyota', 'Volkswagen','Volvo'))
    st.write('You selected:', Make)

    Model = st.selectbox(
            'Select the brand model',
            ('MDX', 'RSX Type S 2dr', 'TSX 4dr', 'TL 4dr', '3.5 RL 4dr',
            '3.5 RL w/Navigation 4dr', 'NSX coupe 2dr manual S', 'A4 1.8T 4dr',
            'A41.8T convertible 2dr', 'A4 3.0 4dr',
           'A4 3.0 Quattro 4dr manual', 'A4 3.0 Quattro 4dr auto',
           'A6 3.0 4dr', 'A6 3.0 Quattro 4dr', 'A4 3.0 convertible 2dr',
           'A4 3.0 Quattro convertible 2dr', 'A6 2.7 Turbo Quattro 4dr',
           'A6 4.2 Quattro 4dr', 'A8 L Quattro 4dr', 'S4 Quattro 4dr',
           'RS 6 4dr', 'TT 1.8 convertible 2dr (coupe)',
           'TT 1.8 Quattro 2dr (convertible)',
           'TT 3.2 coupe 2dr (convertible)', 'A6 3.0 Avant Quattro',
           'S4 Avant Quattro', 'X3 3.0i', 'X5 4.4i', '325i 4dr', '325Ci 2dr',
           '325Ci convertible 2dr', '325xi 4dr', '330i 4dr', '330Ci 2dr',
           '330xi 4dr', '525i 4dr', '330Ci convertible 2dr', '530i 4dr',
           '545iA 4dr', '745i 4dr', '745Li 4dr', 'M3 coupe 2dr',
           'M3 convertible 2dr', 'Z4 convertible 2.5i 2dr',
           'Z4 convertible 3.0i 2dr', '325xi Sport', 'Rainier',
           'Rendezvous CX', 'Century Custom 4dr', 'LeSabre Custom 4dr',
           'Regal LS 4dr', 'Regal GS 4dr', 'LeSabre Limited 4dr',
           'Park Avenue 4dr', 'Park Avenue Ultra 4dr', 'Escalade', 'SRX V8',
           'CTS VVT 4dr', 'Deville 4dr', 'Deville DTS 4dr', 'Seville SLS 4dr',
           'XLR convertible 2dr', 'Escalade EXT', 'Suburban 1500 LT',
           'Tahoe LT', 'TrailBlazer LT', 'Tracker', 'Aveo 4dr',
           'Aveo LS 4dr hatch', 'Cavalier 2dr', 'Cavalier 4dr',
           'Cavalier LS 2dr', 'Impala 4dr', 'Malibu 4dr', 'Malibu LS 4dr',
           'Monte Carlo LS 2dr', 'Impala LS 4dr', 'Impala SS 4dr',
           'Malibu LT 4dr', 'Monte Carlo SS 2dr', 'Astro', 'Venture LS',
           'Corvette 2dr', 'Corvette convertible 2dr', 'Avalanche 1500',
           'Colorado Z85', 'Silverado 1500 Regular Cab', 'Silverado SS',
           'SSR', 'Malibu Maxx LS', 'PT Cruiser 4dr',
           'PT Cruiser Limited 4dr', 'Sebring 4dr', 'Sebring Touring 4dr',
           '300M 4dr', 'Concorde LX 4dr', 'Concorde LXi 4dr',
           'PT Cruiser GT 4dr', 'Sebring convertible 2dr',
           '300M Special Edition 4dr', 'Sebring Limited convertible 2dr',
           'Town and Country LX', 'Town and Country Limited', 'Crossfire 2dr',
           'Pacifica', 'Durango SLT', 'Neon SE 4dr', 'Neon SXT 4dr',
           'Intrepid SE 4dr', 'Stratus SXT 4dr', 'Stratus SE 4dr',
           'Intrepid ES 4dr', 'Caravan SE', 'Grand Caravan SXT',
           'Viper SRT-10 convertible 2dr', 'Dakota Regular Cab',
           'Dakota Club Cab', 'Ram 1500 Regular Cab ST', 'Excursion 6.8 XLT',
           'Expedition 4.6 XLT', 'Explorer XLT V6', 'Escape XLS',
           'Focus ZX3 2dr hatch', 'Focus LX 4dr', 'Focus SE 4dr',
           'Focus ZX5 5dr', 'Focus SVT 2dr', 'Taurus LX 4dr',
           'Taurus SES Duratec 4dr', 'Crown Victoria 4dr',
           'Crown Victoria LX 4dr', 'Crown Victoria LX Sport 4dr',
           'Freestar SE', 'Mustang 2dr (convertible)',
           'Mustang GT Premium convertible 2dr',
           'Thunderbird Deluxe convert w/hardtop 2d', 'F-150 Regular Cab XL',
           'F-150 Supercab Lariat', 'Ranger 2.3 XL Regular Cab', 'Focus ZTW',
           'Taurus SE', 'Envoy XUV SLE', 'Yukon 1500 SLE',
           'Yukon XL 2500 SLT', 'Safari SLE', 'Canyon Z85 SL Regular Cab',
           'Sierra Extended Cab 1500', 'Sierra HD 2500', 'Sonoma Crew Cab',
           'Civic Hybrid 4dr manual (gas/electric)',
           'Insight 2dr (gas/electric)', 'Pilot LX', 'CR-V LX', 'Element LX',
           'Civic DX 2dr', 'Civic HX 2dr', 'Civic LX 4dr', 'Accord LX 2dr',
           'Accord EX 2dr', 'Civic EX 4dr', 'Civic Si 2dr hatch',
           'Accord LX V6 4dr', 'Accord EX V6 2dr', 'Odyssey LX', 'Odyssey EX',
           'S2000 convertible 2dr', 'H2', 'Santa Fe GLS', 'Accent 2dr hatch',
           'Accent GL 4dr', 'Accent GT 2dr hatch', 'Elantra GLS 4dr',
           'Elantra GT 4dr', 'Elantra GT 4dr hatch', 'Sonata GLS 4dr',
           'Sonata LX 4dr', 'XG350 4dr', 'XG350 L 4dr', 'Tiburon GT V6 2dr',
           'G35 4dr', 'G35 Sport Coupe 2dr', 'I35 4dr', 'M45 4dr',
           'Q45 Luxury 4dr', 'FX35', 'FX45', 'Ascender S', 'Rodeo S',
           'X-Type 2.5 4dr', 'X-Type 3.0 4dr', 'S-Type 3.0 4dr',
           'S-Type 4.2 4dr', 'S-Type R 4dr', 'Vanden Plas 4dr', 'XJ8 4dr',
           'XJR 4dr', 'XK8 coupe 2dr', 'XK8 convertible 2dr', 'XKR coupe 2dr',
           'XKR convertible 2dr', 'Grand Cherokee Laredo', 'Liberty Sport',
           'Wrangler Sahara convertible 2dr', 'Sorento LX', 'Optima LX 4dr',
           'Rio 4dr manual', 'Rio 4dr auto', 'Spectra 4dr',
           'Spectra GS 4dr hatch', 'Spectra GSX 4dr hatch',
           'Optima LX V6 4dr', 'Amanti 4dr', 'Sedona LX', 'Rio Cinco',
           'Range Rover HSE', 'Discovery SE', 'Freelander SE', 'GX 470',
           'LX 470', 'RX 330', 'ES 330 4dr', 'IS 300 4dr manual',
           'IS 300 4dr auto', 'GS 300 4dr', 'GS 430 4dr', 'LS 430 4dr',
           'SC 430 convertible 2dr', 'IS 300 SportCross', 'Navigator Luxury',
           'Aviator Ultimate', 'LS V6 Luxury 4dr', 'LS V6 Premium 4dr',
           'LS V8 Sport 4dr', 'LS V8 Ultimate 4dr', 'Town Car Signature 4dr',
           'Town Car Ultimate 4dr', 'Town Car Ultimate L 4dr', 'Cooper',
           'Cooper S', 'Tribute DX 2.0', 'Mazda3 i 4dr', 'Mazda3 s 4dr',
           'Mazda6 i 4dr', 'MPV ES', 'MX-5 Miata convertible 2dr',
           'MX-5 Miata LS convertible 2dr', 'RX-8 4dr automatic',
           'RX-8 4dr manual', 'B2300 SX Regular Cab', 'B4000 SE Cab Plus',
           'G500', 'ML500', 'C230 Sport 2dr', 'C320 Sport 2dr', 'C240 4dr',
           'C320 Sport 4dr', 'C320 4dr', 'C32 AMG 4dr', 'CL500 2dr',
           'CL600 2dr', 'CLK320 coupe 2dr (convertible)',
           'CLK500 coupe 2dr (convertible)', 'E320 4dr', 'E500 4dr',
           'S430 4dr', 'S500 4dr', 'SL500 convertible 2dr', 'SL55 AMG 2dr',
           'SL600 convertible 2dr', 'SLK230 convertible 2dr', 'SLK32 AMG 2dr',
           'C240', 'E320', 'E500', 'Mountaineer', 'Sable GS 4dr',
           'Grand Marquis GS 4dr', 'Grand Marquis LS Premium 4dr',
           'Sable LS Premium 4dr', 'Grand Marquis LS Ultimate 4dr',
           'Marauder 4dr', 'Monterey Luxury', 'Sable GS', 'Endeavor XLS',
           'Montero XLS', 'Outlander LS', 'Lancer ES 4dr', 'Lancer LS 4dr',
           'Galant ES 2.4L 4dr', 'Lancer OZ Rally 4dr auto',
           'Diamante LS 4dr', 'Galant GTS 4dr', 'Eclipse GTS 2dr',
           'Eclipse Spyder GT convertible 2dr', 'Lancer Evolution 4dr',
           'Lancer Sportback LS', 'Pathfinder Armada SE', 'Pathfinder SE',
           'Xterra XE V6', 'Sentra 1.8 4dr', 'Sentra 1.8 S 4dr',
           'Altima S 4dr', 'Sentra SE-R 4dr', 'Altima SE 4dr',
           'Maxima SE 4dr', 'Maxima SL 4dr', 'Quest S', 'Quest SE',
           '350Z coupe 2dr', '350Z Enthusiast convertible 2dr',
           'Frontier King Cab XE V6', 'Titan King Cab XE', 'Murano SL',
           'Alero GX 2dr', 'Alero GLS 2dr', 'Silhouette GL', 'Aztekt',
           'Sunfire 1SA 2dr', 'Grand Am GT 2dr', 'Grand Prix GT1 4dr',
           'Sunfire 1SC 2dr', 'Grand Prix GT2 4dr', 'Bonneville GXP 4dr',
           'Montana', 'Montana EWB', 'GTO 2dr', 'Vibe', 'Cayenne S',
           '911 Carrera convertible 2dr (coupe)',
           '911 Carrera 4S coupe 2dr (convert)', '911 Targa coupe 2dr',
           '911 GT2 2dr', 'Boxster convertible 2dr',
           'Boxster S convertible 2dr', '9-3 Arc Sport 4dr', '9-3 Aero 4dr',
           '9-5 Arc 4dr', '9-5 Aero 4dr', '9-3 Arc convertible 2dr',
           '9-3 Aero convertible 2dr', '9-5 Aero', 'VUE', 'Ion1 4dr',
           'lon2 4dr', 'lon3 4dr', 'lon2 quad coupe 2dr',
           'lon3 quad coupe 2dr', 'L300-2 4dr', 'L300 2', 'xA 4dr hatch',
           'xB', 'Impreza 2.5 RS 4dr', 'Legacy L 4dr', 'Legacy GT 4dr',
           'Outback Limited Sedan 4dr', 'Outback H6 4dr',
           'Outback H-6 VDC 4dr', 'Impreza WRX 4dr', 'Impreza WRX STi 4dr',
           'Baja', 'Forester X', 'Outback', 'XL-7 EX', 'Vitara LX',
           'Aeno S 4dr', 'Aerio LX 4dr', 'Forenza S 4dr', 'Forenza EX 4dr',
           'Verona LX 4dr', 'Aerio SX', 'Prius 4dr (gas/electric)',
           'Sequoia SR5', '4Runner SR5 V6', 'Highlander V6', 'Land Cruiser',
           'RAV4', 'Corolla CE 4dr', 'Corolla S 4dr', 'Corolla LE 4dr',
           'Echo 2dr manual', 'Echo 2dr auto', 'Echo 4dr', 'Camry LE 4dr',
           'Camry LE V6 4dr', 'Camry Solara SE 2dr', 'Camry Solara SE V6 2dr',
           'Avalon XL 4dr', 'Camry XLE V6 4dr', 'Camry Solara SLE V6 2dr',
           'Avalon XLS 4dr', 'Sienna CE', 'Sienna XLE Limited',
           'Celica GT-S 2dr', 'MR2 Spyder convertible 2dr', 'Tacoma',
           'Tundra Regular Cab V6', 'Tundra Access Cab V6 SR5', 'Matrix XR',
           'Touareg V6', 'Golf GLS 4dr', 'GTI 1.8T 2dr hatch',
           'Jetta GLS TDI 4dr', 'New Beetle GLS 1.8T 2dr',
           'Jetta GLI VR6 4dr', 'New Beetle GLS convertible 2dr',
           'Passat GLS 4dr', 'Passat GLX V6 4MOTION 4dr',
           'Passat W8 4MOTION 4dr', 'Phaeton 4dr', 'Phaeton W12 4dr',
           'Jetta GL', 'Passat GLS 1.8T', 'Passat W8', 'XC90 T6', 'S40 4dr',
           'S60 2.5 4dr', 'S60 T5 4dr', 'S60 R 4dr', 'S80 2.9 4dr',
           'S80 2.5T 4dr', 'C70 LPT convertible 2dr',
           'C70 HPT convertible 2dr', 'S80 T6 4dr', 'V40', 'XC70'))
    st.write('You selected:', Model)


    Type = st.selectbox(
           'Select the vehicle type',
           ('SUV', 'Sedan', 'Sports', 'Wagon', 'Truck', 'Hybrid'))
    st.write('You selected:', Type)


    Origin = st.selectbox(
            'Select the car origin area',
             ('Asia', 'Europe', 'USA'))
    st.write('You selected:', Origin)


    DriveTrain = st.selectbox(
                 'Select the DriveTrain Type ',
                 ('All', 'Front', 'Rear'))
    st.write('You selected:', DriveTrain)


    EngineSize = st.number_input("Enter Engine Size ")

    Cylinders = st.number_input("Enter no.of.cylinders ")

    Horsepower = st.number_input("Enter the car HP ")

    MPG_City = st.number_input("Enter MPG in City ")

    MPG_Highway = st.number_input("Enter MPG on Highway ")

    Weight = st.number_input("Enter Weight of the car ")

    Wheelbase = st.number_input("Enter Wheelbase measure ")

    Length = st.number_input("Enter the Car Length ")


    data = {'EngineSize':EngineSize,
            'Cylinders':Cylinders,
            'Horsepower':Horsepower,
            'MPG_City':MPG_City,
            'MPG_Highway':MPG_Highway,
            'Weight':Weight,
            'Wheelbase':Wheelbase,
            'Length':Length,

            'Make':Make,
            'Model':Model,
            'Type':Type,
            'Origin':Origin,
            'DriveTrain':DriveTrain
            }

    features = pd.DataFrame(data,index=[0])
    return features


input_df = user_input_features()

df = pd.read_csv('cars_data.csv')

df=df.dropna()

df['MSRP']=df['MSRP'].str.replace('$','')
df['MSRP']=df['MSRP'].str.replace(',','')
df['MSRP']=df['MSRP'].astype(int)

df.drop(['Invoice'],axis=1,inplace=True)

X = df.drop('MSRP',axis=1)
df = pd.concat([input_df,X],axis=0)

cat_cols = ['Make', 'Model', 'Type', 'Origin', 'DriveTrain']

for col in cat_cols:
    dummy = pd.get_dummies(df[col],prefix=col)
    df=pd.concat([df,dummy],axis=1)
    del df[col]
df=df[:1]

#Write out input selection
#st.subheader('User Input (Pandas DataFrame)')
#st.write(df)

#Loading the model
model = xgboost.XGBRegressor()
model.load_model("XGB_model.bin")

if st.button("Predict"):
    result = prediction(df)
    st.success('Your Car Estimated Price is : $  {}'.format(result))

# defining the function which will make the prediction using
# the data which the user inputs
