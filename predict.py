import pickle
import pandas as pd

model = open("model_pickle", "rb")
model = pickle.load(model)
dt = open("model_pickle_dt", "rb")
dt = pickle.load(dt)
modelb = open("model_pickle_b", "rb")
modelb = pickle.load(modelb)


predictlr= [[
    8500, 920.0, 800.000000, 32.768764, 4, 6, 3.0, 0.0, 1, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

lr = model.predict(predictlr)


p = {"price": [240000.0 ], "surface_of_good": [165.0], "swimming_pool": [1.0], "state_of_building": [4]}
testdata = pd.DataFrame(p)

cl = dt.predict(testdata)


house_data = {'postal_code' :[1060],'property_type' :["HOUSE"],'property_subtype' :["HOUSE"],'type_of_sale':["BUY_REGULAR"],'living_area':["220.0"],'kitchen_type':["SEMI_EQUIPPED"],'fully_equipped_kitchen':[1.0],'open_fire':[0],'terrace':[0],'terrace_area':[50.0],'garden':[1.0],'garden_area':[100.0],'surface_of_good':[218.0],'number_of_facades':[4],'swimming_pool':[1.0],'state_of_building':['GOOD'],'main_city':['wevelgem'], "province":["west-vlaanderen"]}
test_df = pd.DataFrame(house_data)

bpr = modelb.predict(test_df)

print(f"liner regression model:{lr} \n"
      f"discicion Tree model: {cl}\n"
      f"bagging regressor model:{bpr}")