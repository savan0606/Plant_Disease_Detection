#Required Libraries
from flask import Flask, render_template, request
import os
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


#Load Model
path = 'C:/Users/7470/Desktop/Project_Plant_Disease/1649687257.h5'
trainedmodel = tf.keras.models.load_model((path),custom_objects={'KerasLayer':hub.KerasLayer})
print("Model Loaded")

IMAGE_SHAPE = (224, 224)
BATCH_SIZE = 64

data_dir = 'C:/Users/7470/Documents/GitHub/Plant_Disease_Detection/dataset'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
  rescale = 1./255,
  rotation_range=40,
  horizontal_flip=True,
  width_shift_range=0.2, 
  height_shift_range=0.2,
  shear_range=0.2, 
  zoom_range=0.2,
  fill_mode='nearest' )
  
train_generator = train_datagen.flow_from_directory(
    train_dir, 
    subset="training", 
    shuffle=True, 
    seed=42,
    color_mode="rgb", 
    class_mode="categorical",
    target_size=IMAGE_SHAPE,
    batch_size=BATCH_SIZE)

classes = {j: i for i, j in train_generator.class_indices.items()}
print(classes)


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('PlantDiseaseDetection.html')

@app.route('/predict', methods=['POST'])
def predict():
    filename = request.form['filename']
    img = cv2.imread(os.path.join('C:/Users/7470/Desktop/Project_Plant_Disease/TestingImages', filename))
    img = cv2.resize(img, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]) )
    img = img /255
    
    probabilities = trainedmodel.predict(np.asarray([img]))[0]
    class_idx = np.argmax(probabilities)
    pred = classes[class_idx]
    
    if class_idx == 0:
        detail1="apple scab, disease of apple trees caused by the ascomycete fungus Venturia inaequalis. Apple scab is found wherever apples and crabapples are grown but is most severe where spring and summer are cool and moist. The disease can cause high crop losses and is thus of economic import to apple growers."
        detail2="Prevention & Treatment : Prune out branches or infected twigs early in the season. If disease is severe enough to warrant chemical control, choose one of the following fungicides for use on apple trees and crabapple trees: thiophanate-methyl, myclobutanil, a copper fungicide or sulfur." 
        detail3="https://www.britannica.com/science/apple-scab"
        return render_template('diseasesdetails.html',data1=pred,data2=1,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 1:
        detail1="Black rot is occasionally a problem on Minnesota apple trees.This fungal disease causes leaf spot, fruit rot and cankers on branches.Trees are more likely to be infected if they are: (1) Not fully hardy in Minnesota.(2) Infected with fire blight.(3) Stressed by environmental factors like drought. Infected leaves develop frog-eye leaf spot.These are circular spots with purplish or reddish edges and light tan interiors."
        detail2="Prevention & Treatment : Prune out dead or diseased branches.Pick all dried and shriveled fruits remaining on the trees.Remove infected plant material from the area.All infected plant parts should be burned, buried or sent to a municipal composting site.Be sure to remove the stumps of any apple trees you cut down. Dead stumps can be a source of spores." 
        detail3="https://extension.umn.edu/plant-diseases/black-rot-apple"
        return render_template('diseasesdetails.html',data1=pred,data2=1,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 2:
        detail1="Prevention & Treatment : On apple and crab-apple trees, look for pale yellow pinhead sized spots on the upper surface of the leaves shortly after bloom. These gradually enlarge to bright orange-yellow spots which make the disease easy to identify. Orange spots may develop on the fruit as well. Heavily infected leaves may drop prematurely."
        detail2="TreatmentChoose resistant cultivars when available.Rake up and dispose of fallen leaves and other debris from under trees.Remove galls from infected junipers. In some cases, juniper plants should be removed entirely.Apply preventative, disease-fighting fungicides labeled for use on apples weekly, starting with bud break, to protect trees from spores being released by the juniper host. This occurs only once per year, so additional applications after this springtime spread are not necessary." 
        detail3="https://www.planetnatural.com/pest-problem-solver/plant-disease/cedar-apple-rust/"
        return render_template('diseasesdetails.html',data1=pred,data2=1,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 3:
        detail1="Apple Leaf Use : To make apple leaf tea, pick a handful of fresh young apple tree leaves. Rip them up in your hand and toss them into a tea cup. Pour boiling water over the leaves and let steep for 10-15 minutes. Strain the leaves out, and enjoy your cooling herbal apple leaf tea."
        detail2="Apple Use : Apples are an incredibly nutritious fruit that offers multiple health benefits. They're rich in fiber and antioxidants. Eating them is linked to a lower risk of many chronic conditions, including diabetes, heart disease, and cancer. Apples may also promote weight loss and improve gut and brain health." 
        detail3="https://www.healthline.com/nutrition/10-health-benefits-of-apples#TOC_TITLE_HDR_12 and https://coldhardyfruits.com/apple-trees/are-apple-blossoms-and-leaves-edible/"
        return render_template('diseasesdetails.html',data1=pred,data2=0,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 4:
        detail1="Clinical trials have shown that Blueberry Leaf may be capable of benefiting:(1)Lowers blood sugar on an average of 26%.(2)Contains excellent anti-inflammatory properties.(3)Assist and stop damage to the blood vessels sometimes caused by diabetes. Stopping the damage to the blood vessels leading to the retina is one such example.Properly managed blood sugar levels help diabetics ensure a healthy heart and protect against many chronic health conditions such as, neuropathy and retinopathy.(4)Cancer"
        detail2="Blueberry tea is made by steeping leaves of the blueberry bush in hot water. A fragrant and delicious beverage, it provides a number of unique health benefits that make it both refreshing to drink and beneficial to your body." 
        detail3="https://wildaboutberries.com/leaf-benefits/"
        return render_template('diseasesdetails.html',data1=pred,data2=0,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 5:
        detail1="Powdery mildew of sweet and sour cherry is caused by Podosphaera clandestina, an obligate biotrophic fungus. Mid- and late-season sweet cherry (Prunus avium) cultivars are commonly affected, rendering them unmarketable due to the covering of white fungal growth on the cherry surface"
        detail2="IDENTIFICATION : Initial symptoms, often occurring 7 to 10 days after the onset of the first irrigation, are light roughly-circular, powdery looking patches on young, susceptible leaves (newly unfolded, and light green expanding leaves). Older leaves develop an age-related (ontogenic) resistance to powdery mildew and are naturally more resistant to infection than younger leaves." 
        detail3="https://pnwhandbooks.org/plantdisease/host-disease/cherry-prunus-spp-powdery-mildew"
        return render_template('diseasesdetails.html',data1=pred,data2=1,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 6:
        detail1="Benefits of Cherry Leaves for Health : (1) Treat headaches (2) Control blood sugar (3) Overcoming inflammation (4) Maintain heart health (5) Prevent high blood pressure"
        detail2="This plant has a small fruit but tastes very sweet. Apparently, not only the fruit can be consumed, but also the leaves. The many benefits of cherry leaves for health, because they contain vitamins A and C." 
        detail3="https://www.dailyspoilers.com/health/11-benefits-of-cherry-leaves-for-good-health/"
        return render_template('diseasesdetails.html',data1=pred,data2=0,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 7:
        detail1="Grey leaf spot (GLS) is a foliar fungal disease that affects maize, also known as corn. GLS is considered one of the most significant yield-limiting diseases of corn worldwide."
        detail2="There are two fungal pathogens that cause GLS: Cercospora zeae-maydis and Cercospora zeina.Symptoms seen on corn include leaf lesions, discoloration (chlorosis), and foliar blight. Distinct symptoms of GLS are rectangular, brown to gray necrotic lesions that run parallel to the leaf, spanning the spaces between the secondary leaf veins.The fungus survives in the debris of topsoil and infects healthy crops via asexual spores called conidia." 
        detail3="https://en.wikipedia.org/wiki/Corn_grey_leaf_spot"
        return render_template('diseasesdetails.html',data1=pred,data2=1,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 8:
        detail1="Common corn rust, caused by the fungus Puccinia sorghi, is the most frequently occurring of the two primary rust diseases of corn in the U.S., but it rarely causes significant yield losses in Ohio field (dent) corn."
        detail2="Prevention & Treatment : To reduce the incidence of corn rust, plant only corn that has resistance to the fungus. Resistance is either in the form of race-specific resistance or partial rust resistance. In either case, no sweet corn is completely resistant. If the corn begins to show symptoms of infection, immediately spray with a fungicide." 
        detail3="https://www.gardeningknowhow.com/edible/vegetables/corn/corn-rust-fungus-control.htm"
        return render_template('diseasesdetails.html',data1=pred,data2=1,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 9:
        detail1="Northern corn leaf blight (NCLB) or Turcicum leaf blight (TLB) is a foliar disease of corn (maize) caused by Exserohilum turcicum, the anamorph of the ascomycete Setosphaeria turcica. With its characteristic cigar-shaped lesions, this disease can cause significant yield loss in susceptible corn hybrids."
        detail2="Prevention & Treatment : Treating northern corn leaf blight involves using fungicides. For most home gardeners this step isn't needed, but if you have a bad infection, you may want to try this chemical treatment. The infection usually begins around the time of silking, and this is when the fungicide should be applied." 
        detail3="https://www.gardeningknowhow.com/edible/vegetables/corn/northern-corn-leaf-blight-control.htm"
        return render_template('diseasesdetails.html',data1=pred,data2=1,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 10:
        detail1="Corn is rich in vitamin C, an antioxidant that helps protect your cells from damage and wards off diseases like cancer and heart disease. Yellow corn is a good source of the carotenoids lutein and zeaxanthin, which are good for eye health and help prevent the lens damage that leads to cataracts."
        detail2="Use : Corn/Maize Health Benefits:(1) Augments Eye Health.(2) Supplies Essential Amino Acids.(3) Supports A Gluten-Free Diet.(4) Fortifies Bone Density.(5) Keeps Blood Sugar Levels In Check.(6) Treats Anaemia.(7) Boosts Nervous System Function.(8) Augments Heart Health." 
        detail3="https://www.netmeds.com/health-library/post/corn-maize-cholam-health-benefits-nutrition-uses-for-skin-and-hair-recipes-side-effects"
        return render_template('diseasesdetails.html',data1=pred,data2=0,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 11:
        detail1="Grape black rot is a fungal disease caused by an ascomycetous fungus.Relatively small, brown circular lesions develop on infected leaves and within a few days tiny black spherical fruiting bodies (pycnidia) protrude from them."
        detail2="Prevention & Treatment : Remove infected plant material from the vineyard and destroy it. Dormant applications of lime sulfur or Bordeaux mixture are effective against the fungus, as are foliar applications of registered fungicides on two-week intervals during the growing season." 
        detail3="https://www.canr.msu.edu/news/anthracnose_how_to_recognize_and_control_this_fungal_disease_of_grapevines"
        return render_template('diseasesdetails.html',data1=pred,data2=1,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 12:
        detail1="Grapevine measles, also called esca, black measles or Spanish measles, has long plagued grape growers with its cryptic expression of symptoms and, for a long time, a lack of identifiable causal organism(s)."
        detail2="Leaf symptoms are characterized by a ‘tiger stripe’ pattern (Fig 2-bottom leaf) when infections are severe from year to year. Mild infections can produce leaf symptoms that can be confused with other diseases or nutritional deficiencies. White cultivars will display areas of chlorosis followed by necrosis, while red cultivars are characterized by red areas followed by necrosis. Early spring symptoms include shoot tip dieback, leaf discoloration and complete defoliation in severe cases." 
        detail3="https://www2.ipm.ucanr.edu/agriculture/grape/esca-black-measles/"
        return render_template('diseasesdetails.html',data1=pred,data2=1,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 13:
        detail1="The yellow-green disease spots gradually appear on the fronts of the grape leaves with downy mildew, and white frosty mildew appears on the backs of the leaves. Leaf blight produces dark brown patches on the surface of grape leaves."
        detail2="Prevention & Treatment : Spraying of the grapevines at 3-4 leaf stage with fungicides like Bordeaux mixture @ 0.8% or Copper Oxychloride @ 0.25% or Carbendazim @ 0.1% are effective against this disease." 
        detail3="https://en.wikipedia.org/wiki/List_of_grape_diseases"
        return render_template('diseasesdetails.html',data1=pred,data2=1,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 14:
        detail1="For those watching their weight, grape leaves are very low in calories -- about 14 calories for every five leaves. For general health and wellness, grape leaves are a good source of nutrients, including vitamins C, E, A, K and B6, plus niacin, iron, fiber, riboflavin, folate, calcium, magnesium, copper and manganese."
        detail2="Use and benefits : Grape leaves can be used raw in salads or in cooked applications such as steaming and boiling. They are most commonly stuffed with seasonal and regional vegetables, rice, and meats and are cooked into a soft texture. They can also be adorned with traditional sauces made from cheeses, citrus, cream, olive oil, vinegar." 
        detail3="https://bentleysroastbeef.com/benefits-of-eating-grape-leaves-best-restaurant-nashua-nh/"
        return render_template('diseasesdetails.html',data1=pred,data2=0,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 15:
        detail1="Citrus greening is spread by a disease-infected insect, the Asian citrus psyllid (Diaphorina citri Kuwayama or ACP), and has put the future of America's citrus at risk. Infected trees produce fruits that are green, misshapen and bitter, unsuitable for sale as fresh fruit or for juice."
        detail2="Once a tree has citrus greening, there is no cure. Over time, your tree will deteriorate and the disease will ultimately destroy the tree. It is incredibly important to remove trees that have citrus greening disease." 
        detail3="https://www.citrusalert.com/about-citrus-greening/citrus-greening-qa/"
        return render_template('diseasesdetails.html',data1=pred,data2=1,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 16:
        detail1="Bacterial spot is an important disease of peaches, nectarines, apricots, and plums caused by Xanthomonas campestris pv. pruni. Symptoms of this disease include fruit spots, leaf spots, and twig cankers.Peach scab is caused by Cladosporium carpophilum, a fungus that occurs worldwide and affects peach trees in regions with a warm, humid climate conducive to the disease. The pathogen can infect all stone fruits, but is more severe on peaches."
        detail2="A plant with bacterial spot cannot be cured. Remove symptomatic plants from the field or greenhouse to prevent the spread of bacteria to healthy plants. Burn, bury or hot compost the affected plants and DO NOT eat symptomatic fruit." 
        detail3="https://www.gardeningknowhow.com/edible/fruits/peach/bacterial-spot-on-peach-trees.htm"
        return render_template('diseasesdetails.html',data1=pred,data2=1,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 17:
        detail1="Medicinal uses of the Peach leaf of the peach include treatment for fever, headache, eczema, ulcers and malaria. The gummy of the peach tree has a bitter taste but is used as a cure for quenching thirst by boiling the gummy in water. It is also used for treating diabetes, dysentery and urolithiasis."
        detail2="Peach tea is tea made with leaves or fruit of peach tree – Prunus persica. Tea made from peach leaves is still a very uncommon beverage and usually prepared for its potential health benefits. Peach fruit tea, on the other hand, is one of the most popular teas in the world. It can be prepared from fruits only or blended with other teas. The most popular blend is peach black tea. Because of its refreshing, sweet and strong flavor, this tea is often served iced." 
        detail3="https://www.google.com/search?q=peach+leaf+uses&oq=p&aqs=chrome.0.69i59l3j69i57j0i67l2j0i131i433i512j46i67j0i67l2.1072j0j15&sourceid=chrome&ie=UTF-8" 
        return render_template('diseasesdetails.html',data1=pred,data2=0,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 18:
        detail1="Bacterial leaf spot, caused by Xanthomonas campestris pv. vesicatoria, is the most common and destructive disease for peppers in the eastern United States. It is a gram-negative, rod-shaped bacterium that can survive in seeds and plant debris from one season to another (Frank et al."
        detail2="Prevention & Treatment : Copper-containing bactericides provide a protective cover on foliage and fruit. Bacterial viruses (bacteriophages) that specifically kill the bacteria are available. Submerge seeds for one minute in 1.3% sodium hypochlorite or in hot water (50°C) for 25 minutes." 
        detail3="https://www.apsnet.org/edcenter/disandpath/prokaryote/pdlessons/Pages/Bacterialspot.aspx"
        return render_template('diseasesdetails.html',data1=pred,data2=1,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 19:
        detail1="Benefits : Bell peppers are rich in antioxidants, which are associated with better health and protection against conditions like heart disease and cancer. For instance, peppers are especially rich in antioxidant vitamins including vitamins C, E and beta-carotene."
        detail2="Red peppers : Red peppers pack the most nutrition, because they've been on the vine longest. Green peppers are harvested earlier, before they have a chance to turn yellow, orange, and then red. Compared to green bell peppers, the red ones have almost 11 times more beta-carotene and 1.5 times more vitamin C." 
        detail3="https://www.bbcgoodfood.com/howto/guide/top-5-health-benefits-of-peppers"
        return render_template('diseasesdetails.html',data1=pred,data2=0,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 20:
        detail1="Early Blight in Potato (PP1892, June 2018) Early blight of potato is caused by the fungal pathogen Alternaria solani. The disease affects leaves, stems and tubers and can reduce yield, tuber size, storability of tubers, quality of fresh-market and processing tubers and marketability of the crop."
        detail2="Symptoms of Potatoes with Early Blight Early blight rarely affects young plants. Symptoms first occur on the lower or oldest leaves of the plant. Dark, brown spots appear on this older foliage and, as the disease progresses, enlarge, taking on an angular shape. These lesions often look like a target and, in fact, the disease is sometimes referred to as target spot. As the spots enlarge, they may cause the entire leaf to yellow and die, but remain on the plant. Dark brown to black spots may also occur on the stems of the plant." 
        detail3="Treatment : https://www.gardeningknowhow.com/edible/vegetables/potato/potato-early-blight-treatment.htm"
        return render_template('diseasesdetails.html',data1=pred,data2=1,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 21:
        detail1="late blight, also called potato blight, disease of potato and tomato plants that is caused by the water mold Phytophthora infestans. The disease occurs in humid regions with temperatures ranging between 4 and 29 °C (40 and 80 °F). Hot dry weather checks its spread."
        detail2="Prevention & Treatment : Late blight is controlled by eliminating cull piles and volunteer potatoes, using proper harvesting and storage practices, and applying fungicides when necessary. Air drainage to facilitate the drying of foliage each day is important." 
        detail3="https://www.intechopen.com/chapters/58251"
        return render_template('diseasesdetails.html',data1=pred,data2=1,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 22:
        detail1="Use : 1. Raw: just like any dark leafy green you can add them to your salads 2. Sauteed: roughly chop them up and sauté them with some butter and garlic 3. Boiled: boiling sweet potato vine leaves will help remove their bitterness."
        detail2="Not only are sweet potato vines low in calories with just 12 calories per cup, but they are a great source of numerous fibre, antioxidants, essential vitamins A, B, C, D, E and K and minerals like niacin, thiamine and beta carotene. " 
        detail3="https://www.newidea.com.au/sweet-potato-leaves"
        return render_template('diseasesdetails.html',data1=pred,data2=0,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 23:
        detail1="Red raspberry leaf has been recommended as a tonic to improve fat metabolism and encourage weight loss. It is often sold as a “detoxifying” supplement meant to improve body composition and overall health."
        detail2="Good Source of Nutrients and Antioxidants : Red raspberry leaves are rich in vitamins and minerals. They provide B vitamins, vitamin C and a number of minerals, including potassium, magnesium, zinc, phosphorus and iron. However, their most notable contribution might be their antioxidant properties (1, 2 )." 
        detail3="https://www.healthline.com/nutrition/red-raspberry-leaf-tea"
        return render_template('diseasesdetails.html',data1=pred,data2=0,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 24:
        detail1="Fresh Green Soybean Leaves.The most common ways of cooking them are to stir fry with sliced pork or chicken, or to simmer in seasoned broth (Wu 1848)."
        detail2="Here are the potential health benefits of soybeans.(1)Soybean Helps Relieve Sleep Disorders.(2)Soybean May Help Manage Diabetes.(3)Soybean Help Improve Blood Circulation.(4)Soybean Essential for Pregnancy.(5)Soybean for Healthy Bones. (6)Soybean Aids Healthy Digestion. (7)Relieve Menopausal Symptoms.(8)Soybean Improves Heart Health."
        detail3="https://www.soyinfocenter.com/HSS/green_vegetable.php"
        return render_template('diseasesdetails.html',data1=pred,data2=0,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 25:
        detail1="Early signs of disease on squash leavesPowdery mildew is most commonly seen on the top of the leaves, but it can also appear on the leaf undersides, the stems, and even on the fruits. Early signs of powdery mildew are small, random patches of white “dust” on the upper leaf surface."
        detail2="Prevention & Treatment : A better treatment solution for your squash plants is baking soda. Baking soda is an excellent option for treating powdery mildew. It is readily available in your home and will not cause any harm to the surrounding vegetable plants. With the baking soda method, you will also need some cooking oil and some dish soap." 
        detail3="https://pinchofseeds.com/powdery-mildew-on-plants/"
        return render_template('diseasesdetails.html',data1=pred,data2=1,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 26:
        detail1="Scorched strawberry leaves are caused by a fungal infection which affects the foliage of strawberry plantings. The fungus responsible is called Diplocarpon earliana. Strawberries with leaf scorch may first show signs of issue with the development of small purplish blemishes that occur on the topside of leaves."
        detail2="Prevention & Treatment : While leaf scorch on strawberry plants can be frustrating, there are some strategies which home gardeners may employ to help prevent its spread in the garden. The primary means of strawberry leaf scorch control should always be prevention." 
        detail3="https://www.gardeningknowhow.com/edible/fruits/strawberry/strawberries-with-leaf-scorch.htm"
        return render_template('diseasesdetails.html',data1=pred,data2=1,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 27:
        detail1="According to the University of Maryland Medical Center, strawberry leaves are high in vitamin C, iron, and calcium, as well as contain tannins, which helps with digestion, nausea, and stomach cramps."
        detail2="A popular summer fruit, strawberries aren't just tasty, they're also healthy. A quick search online will tell you that, yes, strawberry leaves are safe to consume. Strawberry leaves are known for helping with arthritis pain, because they contain a diuretic called caffeic acid. In plain terms, this means it helps relieve water tension from the joints. By reducing inflammation, this will ease any discomfort you may be feeling." 
        detail3="https://spoonuniversity.com/lifestyle/can-you-eat-strawberry-leaves"
        return render_template('diseasesdetails.html',data1=pred,data2=0,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 28:
        detail1="Bacterial spot of tomato is a potentially devastating disease that, in severe cases, can lead to unmarketable fruit and even plant death.  Bacterial spot can occur wherever tomatoes are grown, but is found most frequently in warm, wet climates, as well as in greenhouses.  The disease is often an issue in Wisconsin."
        detail2="A plant with bacterial spot cannot be cured. Remove symptomatic plants from the field or greenhouse to prevent the spread of bacteria to healthy plants. Burn, bury or hot compost the affected plants and DO NOT eat symptomatic fruit." 
        detail3="https://www.gardeningknowhow.com/edible/vegetables/tomato/tomatoes-with-bacterial-canker.htm"
        return render_template('diseasesdetails.html',data1=pred,data2=1,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 29:
        detail1="Early blight can be caused by two closely related species: Alternaria tomatophila and Alternaria solani. The early blight pathogens both overwinter in infected plant debris and soil in Minnesota. The pathogen also survives on tomato seed or may be introduced on tomato transplants."
        detail2="Treatment : Tomatoes that have early blight require immediate attention before the disease takes over the plants. Thoroughly spray the plant (bottoms of leaves also) with Bonide Liquid Copper Fungicide concentrate or Bonide Tomato & Vegetable. Both of these treatments are organic." 
        detail3="https://www.pesches.com/blogs/news/how-to-fight-early-blight"
        return render_template('diseasesdetails.html',data1=pred,data2=1,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 30:
        detail1="Late blight is a potentially devastating disease of tomato and potato, infecting leaves, stems, tomato fruit, and potato tubers. The disease spreads quickly in fields and can result in total crop failure if untreated. Late blight does not occur every year in Minnesota."
        detail2="Tomato late blight is caused by the oomycete pathogen Phytophthora infestans (P. infestans). The pathogen is best known for causing the devastating Irish potato famine of the 1840s, which killed over a million people, and caused another million to leave the country." 
        detail3="https://content.ces.ncsu.edu/tomato-late-blight"
        return render_template('diseasesdetails.html',data1=pred,data2=1,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 31:
        detail1="Tomato leaf mold is a fungal disease that can develop when there are extended periods of leaf wetness and the relative humidity is high (greater than 85 percent). Due to this moisture requirement, the disease is seen primarily in hoophouses and greenhouses."
        detail2="Prevention & Treatment : Use drip irrigation and avoid watering foliage. Use a stake, strings, or prune the plant to keep it upstanding and increase airflow in and around it. Remove and destroy (burn) all plants debris after the harvest." 
        detail3="https://www.gardeningchannel.com/tomato-diseases-how-to-fight-leaf-mold/"
        return render_template('diseasesdetails.html',data1=pred,data2=1,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 32:
        detail1="Septoria leaf spot is caused by a fungus, Septoria lycopersici. It is one of the most destructive diseases of tomato foliage and is particularly severe in areas where wet, humid weather persists for extended periods. Septoria leaf spot usually appears on the lower leaves after the first fruit sets.But you can eat tomatoes with Septoria leaf spot if the disease doesn't affect the whole fruit. Unfortunately, Septoria spreads fast and weakens plant structure within a short time. If whole fruits are heavily affected, it is better not to eat them, especially if you have any physical issues."
        detail2="Prevention & Treatment : (1)Removing infected leaves. Remove infected leaves immediately, and be sure to wash your hands and pruners thoroughly before working with uninfected plants.(2)Consider organic fungicide options.(3)Consider chemical fungicides." 
        detail3="https://www.thespruce.com/identifying-and-controlling-septoria-leaf-spot-of-tomato-1402974"
        return render_template('diseasesdetails.html',data1=pred,data2=1,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 33:
        detail1="The two-spotted spider mite is the most common mite species that attacks vegetable and fruit crops in New England. Spider mites can occur in tomato, eggplant, potato, vine crops such as melons, cucumbers, and other crops. Two-spotted spider mites are one of the most important pests of eggplant."
        detail2="How do you get rid of two spot spider mites? -> Image result for tomato Spider Mites Two-spotted spider miteLong-lasting insecticides, such as bifenthrin and permethrin can be used on twospotted spider infestations. However, these insecticides also kill natural enemies and could possibly make infestations worse in the long run. Twospotted spider mite infestations occur when it is hot and dry." 
        detail3="https://extension.umn.edu/yard-and-garden-insects/spider-mites"
        return render_template('diseasesdetails.html',data1=pred,data2=1,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 34:
        detail1="Target spot of tomato often causes necrotic lesions in a concentric pattern, similar to early blight. Target spot of tomato is favored by temperatures of 68 to 82°F and leaf wetness periods as long as 16 hours. The target spot fungus can survive in host residue for a period."
        detail2="Controlling target spot: (1)practise crop rotation to help reduce initial levels of the disease.(2)plant healthy seed to avoid infecting a clean paddock.(3)keep plants growing vigorously – plants with adequate nutrition and water that are free from other diseases are less prone to infection.(4)apply registered fungicides." 
        detail3="https://agriculture.vic.gov.au/biosecurity/plant-diseases/vegetable-diseases/target-spot-early-blight-of-potatoes"
        return render_template('diseasesdetails.html',data1=pred,data2=1,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 35:
        detail1="Tomato yellow leaf curl virus is a species in the genus Begomovirus and family Geminiviridae. Tomato yellow leaf curl virus (TYLCV) infection induces severe symptoms on tomato plants and causes serious yield losses worldwide. TYLCV is persistently transmitted by the sweetpotato whitefly, Bemisia tabaci (Gennadius)."
        detail2="Use a neonicotinoid insecticide, such as dinotefuran (Venom) imidacloprid (AdmirePro, Alias, Nuprid, Widow, and others) or thiamethoxam (Platinum), as a soil application or through the drip irrigation system at transplanting of tomatoes or peppers." 
        detail3="https://www.farmprogress.com/controlling-tomato-yellow-leaf-curl-virus"
        return render_template('diseasesdetails.html',data1=pred,data2=1,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 36:
        detail1="Tomato mosaic virus (ToMV) is a plant pathogenic virus. It is found worldwide and affects tomatoes and many other plants."
        detail2="Once plants are infected, there is no cure for mosaic viruses. Because of this, prevention is key! However, if plants in your garden do show symptoms of having mosaic viruses, here's how to minimize the damage: Remove all infected plants and destroy them." 
        detail3="https://www.planetnatural.com/pest-problem-solver/plant-disease/mosaic-virus/"
        return render_template('diseasesdetails.html',data1=pred,data2=1,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    elif class_idx == 37:
        detail1="Another study has shown tomato leaves and tomato stems to have higher antioxidant activity and polyphenols (plant-based micronutrients that help fight disease and improve overall health) than tomato fruits. What's most surprising is the discovery of tomatine as a cancer inhibitor."
        detail2="Tomato leaves medicinal uses : Tomato is a plant. The fruit is a familiar vegetable, but the fruit, leaf, and vine are used to make medicine. Tomato is used for preventing cancer of the breast, bladder, cervix, colon and rectum, stomach, lung, ovaries, pancreas, and prostate." 
        detail3="https://www.emedicinehealth.com/tomato/vitamins-supplements.htm"
        return render_template('diseasesdetails.html',data1=pred,data2=0,data3=filename,data4=detail1,data5=detail2,data6=detail3)
    else:
        return render_template('PlantDiseaseDetection.html')

if __name__ == "__main__":
    app.run(debug=True)