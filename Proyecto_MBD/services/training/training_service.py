from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

class TrainingService:
    def train_random_forest(self, X_train, y_train):
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model

    def train_xgboost(self, X_train, y_train, num_classes: int):
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob" if num_classes > 2 else "binary:logistic",
            num_class=num_classes if num_classes > 2 else None,
            eval_metric="mlogloss" if num_classes > 2 else "logloss"
        )
        model.fit(X_train, y_train)
        return model

    def train_lightgbm(self, X_train, y_train, num_classes: int):
        if num_classes > 2:
            model = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=num_classes,
                n_estimators=100,
                learning_rate=0.1
            )
        else:
            model = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=100,
                learning_rate=0.1
            )

        model.fit(X_train, y_train)
        return model