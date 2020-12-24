from sklearn.model_selection import train_test_split

tr_x, va_x, tr_y, va_y = train_test_split(
    train_x, train_y, test_size=0.2, random_state=71, shuffle=True)
