'set working directory
'cd "\\fileserv\FMD\MOST\02 Liquidity Forecasting\CurrencyForecast\New Model\CIC Forecast 2022"

cd "Z:\MOST\02 Liquidity Forecasting\CurrencyForecast\New Model\CIC Forecast 2022"

' Open existing workfile
wfopen "CIC.wf1"

'import rawdata from DummyCICForecast.xlsx
import(mode=o) "DummyCICForecast.xlsx" range="DATA change"!$E$2:$BS$10000 colhead=1 namepos=custom colheadnames=("Name") na="#N/A" @smpl @all

' group rawdata
group group_rawdata Currency	Change	Day	Month	Year	WeekNum_M	WeekNum_Y 	WeekDay	Date_01	 Date_02	Date_03	 Date_04	Date_05	 Date_06 Date_07	Date_08 	Date_09	 Date_10	Date_11	 Date_12	Date_13 	Date_14 	Date_15 	Date_16	 Date_17	Date_18 	Date_19 	Date_20 	Date_21 Date_22 	Date_23 	Date_24 	Date_25	 Date_26	Date_27	 Date_28	Date_29 	Date_30	 Date_31	D_MON	D_TUE	D_WED	D_THU	D_FRI	D_JAN	D_FEB	D_MAR	D_APR	D_MAY	D_JUN	D_JUL	D_AUG	D_SEP	D_OCT	D_NOV	D_DEC	D_WEEK1	D_WEEK2	D_WEEK3	D_WEEK4	D_WEEK5	D_LWD	D_PRE_LH1	D_PRE_LH3	D_POST_LH3	D_PRE_SH1	D_Covid_1st

'set sample period
smpl 4740 9999
copy new_model_master new_model_tempt

' Run LS model with ARMA (ensure correct variable names)
new_model_tempt.ls(optmethod=opg) change c date_02 date_03 date_04 date_05 date_06 date_07 date_08 date_09 date_10 date_11 date_12 date_13 date_14 date_15 date_16 date_17 date_18 date_19 date_20 date_21 date_22 date_23 date_24 date_25 date_26 date_27 date_28 date_29 date_30 date_31 d_tue d_wed d_thu d_fri d_week2 d_week3 d_week4 d_week5 d_jan d_feb d_mar d_apr d_may d_jun d_jul d_aug d_sep d_oct d_nov d_pre_lh3 d_post_lh3 d_pre_sh1 d_covid_1st d_lwd d_pre_lh1 ar(1) ma(1)

'forecast
new_model_tempt.forecast(e, g) changef

' Define today's date and save equation and series
!today = @dateval(@date)
%newname_eq = "new_model_" + @datestr(!today, "YYYYMMDD")
%newname_change = "changef_" + @datestr(!today, "YYYYMMDD")

' save equation / series (delete if exists)
if @isobject(%newname_eq ) then
    delete %newname_eq
endif

if @isobject(%newname_change) then
    delete %newname_change
endif

rename new_model_tempt  {%newname_eq }
rename changef  {%newname_change }

' Save workfile and programe
wfsave(2) CIC


