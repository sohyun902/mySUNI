from steps.step3_recommend import find_similars, find_competitors
from steps.step4_attractiveness import predict_attractiveness
import pandas as pd

def main():
    print("=== main 시작 ===")
    time.sleep(2)

    file_path = "./data/영화DB(임시).csv"
    print(f"파일 읽기: {file_path}")
    df = pd.read_csv(file_path)
    print(f"읽은 데이터 shape: {df.shape}")

    df['유사작'] = find_similars(df)
    print("유사작 계산 완료")

    df['경쟁작'] = find_competitors(df)
    print("경쟁작 계산 완료")

    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print("CSV 저장 완료")

    df['예측 매력도'] = predict_attractiveness(df,
        encoder_path = './steps/ordinal_encoder.pkl',
        model_path = './steps/rf_weighted_model.pkl')
    print("예측 매력도 계산 완료")

    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print("=== main 끝 ===")
if __name__ == "__main__":
    main()
