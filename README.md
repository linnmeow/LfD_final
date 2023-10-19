# LfD_final

Linnnn 17.10: added data visualizer class, rn it doesnt do much and the results of the visualizations do not say much (a lot of "user"). But good to know is that we have 8000/4000 not offensive (labeled as NOT) and offensive tweets (labeled as OFF) in our training set respectively. And 700/350 in dev set.

I need to do preprocessing tmw and prob add more visulization tools. But it was a good start for meeeeeee.


Linnnn 18.10: added data preprocessor (removes @USER and URL, punctuation, remove numbers, convert emojis to text)

original data vs. clean data:

@USER She should ask a few native Americans what their take on this is.	OFF
@USER @USER Go home you’re drunk!!! @USER #MAGA #Trump2020 👊🇺🇸👊 URL	OFF
Amazon is investigating Chinese employees who are selling internal data to third-party sellers looking for an edge in the competitive marketplace. URL #Amazon #MAGA #KAG #CHINA #TCOT	NOT
"@USER Someone should'veTaken"" this piece of shit to a volcano. 😂"""	OFF
@USER @USER Obama wanted liberals &amp; illegals to move into red states	NOT
@USER Liberals are all Kookoo !!!	OFF
@USER @USER Oh noes! Tough shit.	OFF


she should ask a few native americans what their take on this is	OFF
go home youre drunk maga trump oncomingfist UnitedStates oncomingfist	OFF
amazon is investigating chinese employees who are selling internal data to thirdparty sellers looking for an edge in the competitive marketplace amazon maga kag china tcot	NOT
someone shouldvetaken this piece of shit to a volcano facewithtearsofjoy	OFF
obama wanted liberals amp illegals to move into red states	NOT
liberals are all kookoo	OFF
oh noes tough shit	OFF

Linnnn 19.10:
added train and dev datasets with sentiment labels (neg, pos, neu)
added sentiment_analyzer.py that performs this
