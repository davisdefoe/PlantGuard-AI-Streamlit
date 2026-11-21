from pathlib import Path
from datetime import datetime, timezone
import json
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA=Path("dataset"); OUT=Path("models"); OUT.mkdir(exist_ok=True)
SIZE=(224,224); BATCH=32
train=tf.keras.utils.image_dataset_from_directory(DATA/"train",image_size=SIZE,batch_size=BATCH)
valid=tf.keras.utils.image_dataset_from_directory(DATA/"validation",image_size=SIZE,batch_size=BATCH,shuffle=False)
test=tf.keras.utils.image_dataset_from_directory(DATA/"test",image_size=SIZE,batch_size=BATCH,shuffle=False)
names=train.class_names; n=len(names)
if n<2: raise ValueError("Create at least two class folders.")
auto=tf.data.AUTOTUNE
train=train.cache().shuffle(1000).prefetch(auto); valid=valid.cache().prefetch(auto); test=test.cache().prefetch(auto)
aug=tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"),tf.keras.layers.RandomRotation(.08),tf.keras.layers.RandomZoom(.1)])
base=tf.keras.applications.MobileNetV2(input_shape=SIZE+(3,),include_top=False,weights="imagenet")
base.trainable=False
inp=tf.keras.Input(shape=SIZE+(3,)); x=aug(inp); x=tf.keras.applications.mobilenet_v2.preprocess_input(x)
x=base(x,training=False); x=tf.keras.layers.GlobalAveragePooling2D()(x); x=tf.keras.layers.Dropout(.25)(x)
out=tf.keras.layers.Dense(n,activation="softmax")(x); model=tf.keras.Model(inp,out)
callbacks=[tf.keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True,monitor="val_accuracy"),
tf.keras.callbacks.ModelCheckpoint(str(OUT/"best_model.keras"),save_best_only=True,monitor="val_accuracy")]
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(train,validation_data=valid,epochs=5,callbacks=callbacks)
base.trainable=True
for layer in base.layers[:-30]: layer.trainable=False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(train,validation_data=valid,epochs=5,callbacks=callbacks)
model=tf.keras.models.load_model(OUT/"best_model.keras")
true=[]; pred=[]
for images,labels in test:
    p=model.predict(images,verbose=0); true.extend(labels.numpy().tolist()); pred.extend(np.argmax(p,axis=1).tolist())
report={"trained_at":datetime.now(timezone.utc).isoformat(),"architecture":"MobileNetV2 transfer learning + fine-tuning",
"num_classes":n,"class_names":names,"test_accuracy":float(accuracy_score(true,pred)),
"classification_report":classification_report(true,pred,target_names=names,output_dict=True,zero_division=0),
"confusion_matrix":confusion_matrix(true,pred).tolist()}
model.save(OUT/"plantguard_model.keras")
(OUT/"labels.json").write_text(json.dumps(names,indent=2))
(OUT/"training_report.json").write_text(json.dumps(report,indent=2))
print("Done. Exported model, labels and test metrics to models/.")
