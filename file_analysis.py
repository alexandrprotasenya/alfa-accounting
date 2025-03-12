# Standard library imports
import calendar
import io
import logging
from datetime import datetime
from typing import Optional, Tuple

# Third-party imports
import matplotlib.pyplot as plt
# numpy is not needed, removing unused import
import pandas as pd
import streamlit as st

# Constants
CURRENCY = "BYN"
DATE_FORMAT = "%d.%m.%Y"
MONTH_YEAR_FORMAT = "%B %Y"

# Column names
COL_DATE = "Дата"
COL_DESCRIPTION = "Примечание"
COL_AMOUNT = "Сумма в валюте счета"
COL_MONTH = "Месяц"
COL_YEAR = "Год"
COL_MONTH_YEAR = "Месяц-Год"
COL_CATEGORY = "Категория"
COL_TYPE = "Тип"

# Display column names
DISP_DATE = "Дата"
DISP_CATEGORY = "Категория"
DISP_DESCRIPTION = "Описание"
DISP_AMOUNT = "Сумма (BYN)"
DISP_TYPE = "Тип операции"

# Category types
INCOME_TYPE = "Доход"
EXPENSE_TYPE = "Расход"

# Category keywords mapping
CATEGORY_KEYWORDS = {
    'Продукты': ['food', 'магазин', 'shop', 'продукт', 'fudbalans', 'market', 'sem pyatnits'],
    'Транспорт': ['такси', 'taxi', 'uber', 'яндекс', 'yandex', 'транспорт', 'metro', 'метро'],
    'Рестораны': ['кафе', 'ресторан', 'restaurant', 'cafe', 'papa doner', 'coffee', 'кофе'],
    'Здравоохранение': ['аптека', 'pharmacy', 'hospital', 'больница', 'clinic', 'клиника', 'здоровье', 'mayak zdorovya'],
    'Коммунальные': ['комиссия', 'комунальные', 'erip', 'insync', 'услуги'],
    'Покупки': ['mila', 'megatop', 'магазин', 'artagor', 'dm expobel'],
    'Переводы': ['перевод', 'transfer', 'exchange'],
    'Корона': ['universam'],
}

# Default category
DEFAULT_CATEGORY = "Другое"

def extract_category(description: str) -> str:
    """Extract category from transaction description based on keywords.
    
    Args:
        description: The transaction description text
        
    Returns:
        The matched category name or default category if no match is found
    """
    description = description.lower()
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword.lower() in description for keyword in keywords):
            return category
    
    return DEFAULT_CATEGORY

def parse_date(date_str: str) -> Optional[datetime]:
    """Parse date from the format in the CSV file.
    
    Args:
        date_str: Date string in DD.MM.YYYY format
        
    Returns:
        Datetime object if parsing successful, None otherwise
    """
    try:
        return datetime.strptime(date_str, DATE_FORMAT)
    except ValueError:
        return None

def format_currency(val: float) -> str:
    """Format currency values with appropriate formatting.
    
    Args:
        val: The currency value to format
        
    Returns:
        Formatted currency string with 2 decimal places
    """
    return f"{val:.2f} {CURRENCY}"

def process_file(uploaded_file) -> Optional[Tuple[pd.DataFrame, str]]:
    """Process the uploaded CSV file with cp1251 encoding.
    
    Args:
        uploaded_file: File object from Streamlit file uploader
        
    Returns:
        Tuple containing processed dataframe and header information if successful,
        None otherwise
    """
    content = uploaded_file.read()
    
    # Try to decode with cp1251 encoding
    try:
        text = content.decode('cp1251')
    except UnicodeDecodeError:
        st.error("Ошибка декодирования файла с кодировкой cp1251. Пожалуйста, проверьте формат файла.")
        return None
    
    return _parse_csv_content(text)

def _parse_csv_content(text: str) -> Optional[Tuple[pd.DataFrame, str]]:
    """Parse CSV content and create processed dataframe.
    
    Args:
        text: Decoded file content as string
        
    Returns:
        Tuple containing processed dataframe and header information if successful,
        None otherwise
    """
    try:
        # Split lines and validate
        lines = text.split('\n')
        if len(lines) < 3:
            st.error("Файл не содержит достаточное количество строк.")
            return None
        
        # Extract header information for display
        header_info = lines[0]
        
        # Parse CSV data
        csv_data = io.StringIO('\n'.join(lines[1:]))
        df = pd.read_csv(csv_data, delimiter=';')
        
        # Validate required columns
        if not all(col in df.columns for col in [COL_DATE, COL_DESCRIPTION, COL_AMOUNT]):
            st.error("Необходимые столбцы не найдены в CSV файле.")
            return None
        
        # Process the dataframe
        processed_df = _prepare_dataframe(df)
        return processed_df, header_info
        
    except Exception as e:
        logging.error(f"Error processing the file: {str(e)}", exc_info=True)
        st.error(f"Ошибка обработки файла: {str(e)}")
        return None

def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare and enrich the dataframe with derived columns.
    
    Args:
        df: Original dataframe from CSV
        
    Returns:
        Processed dataframe with additional columns
    """
    # Convert date column
    df[COL_DATE] = df[COL_DATE].apply(parse_date)
    
    # Add month and year columns
    df[COL_MONTH] = df[COL_DATE].apply(lambda x: x.month if x else None)
    df[COL_YEAR] = df[COL_DATE].apply(lambda x: x.year if x else None)
    
    # Add month-year column for grouping
    df[COL_MONTH_YEAR] = df.apply(
        lambda x: f"{calendar.month_name[x[COL_MONTH]]} {x[COL_YEAR]}" 
        if x[COL_MONTH] and x[COL_YEAR] else None, 
        axis=1
    )
    
    # Convert amount column to numeric
    df[COL_AMOUNT] = df[COL_AMOUNT].astype(float)
    
    # Add category based on description
    df[COL_CATEGORY] = df[COL_DESCRIPTION].apply(extract_category)
    
    return df

def display_yearly_summary(df: pd.DataFrame) -> None:
    """Display yearly summary of financial data.
    
    Args:
        df: DataFrame containing transaction data
    """
    if df is None or df.empty:
        return
    
    st.subheader("Годовая сводка")
    
    # Calculate and display key financial metrics
    _display_financial_summary(df)
    
    # Display monthly trends chart
    _display_monthly_trends(df)
    
    # Category breakdown section
    st.subheader("Разбивка по категориям за весь период")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Expenses by category
        st.subheader("Расходы по категориям")
        _display_expense_categories(df)
    
    with col2:
        # Income by category
        st.subheader("Доходы по категориям")
        _display_income_categories(df)


def _display_financial_summary(df: pd.DataFrame) -> None:
    """Calculate and display financial summary metrics.
    
    Args:
        df: DataFrame containing transaction data
    """
    # Calculate total income and expenses
    income = df[df[COL_AMOUNT] > 0][COL_AMOUNT].sum()
    expenses = df[df[COL_AMOUNT] < 0][COL_AMOUNT].sum()
    balance = income + expenses  # expenses are negative, so we add them
    
    # Display summary metrics in 3 columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Общие расходы", f"{expenses:.2f} {CURRENCY}", delta=None, delta_color="off")
    with col2:
        st.metric("Общий доход", f"{income:.2f} {CURRENCY}", delta=None, delta_color="off")
    with col3:
        st.metric("Баланс", f"{balance:.2f} {CURRENCY}", delta=f"{balance:.2f}", delta_color="inverse")


def _display_monthly_trends(df: pd.DataFrame) -> None:
    """Display monthly income and expense trends.
    
    Args:
        df: DataFrame containing transaction data
    """
    st.subheader("Ежемесячные тренды")
    
    # Group by month and calculate sums
    monthly_data = df.groupby(COL_MONTH_YEAR).agg(
        Расходы=(COL_AMOUNT, lambda x: x[x < 0].sum()),
        Доходы=(COL_AMOUNT, lambda x: x[x > 0].sum())
    ).reset_index()
    
    # Sort by date
    monthly_data = monthly_data.sort_values(COL_MONTH_YEAR, key=lambda x: pd.to_datetime(x, format=MONTH_YEAR_FORMAT))
    
    # Plot monthly trends
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(monthly_data[COL_MONTH_YEAR]))
    ax.bar(x, monthly_data['Расходы'].abs(), label='Расходы', color='salmon')
    ax.bar(x, monthly_data['Доходы'], label='Доходы', color='skyblue')
    ax.set_xticks(x)
    ax.set_xticklabels(monthly_data[COL_MONTH_YEAR], rotation=45)
    ax.legend()
    ax.set_title('Ежемесячный обзор доходов и расходов')
    ax.set_ylabel(f'Сумма ({CURRENCY})')
    st.pyplot(fig)


def _display_expense_categories(df: pd.DataFrame) -> None:
    """Display expense breakdown by category.
    
    Args:
        df: DataFrame containing transaction data
    """
    # Group by category and calculate expenses
    expense_categories = df[df[COL_AMOUNT] < 0].groupby(COL_CATEGORY)[COL_AMOUNT].sum().reset_index()
    expense_categories = expense_categories.sort_values(COL_AMOUNT)
    
    if not expense_categories.empty:
        # Create pie chart
        fig, ax = plt.subplots(figsize=(10, 6))
        expense_categories['Абс'] = expense_categories[COL_AMOUNT].abs()
        ax.pie(expense_categories['Абс'], labels=expense_categories[COL_CATEGORY], autopct='%1.1f%%')
        ax.set_title('Расходы по категориям')
        st.pyplot(fig)
        
        # Display expense table
        expense_display = expense_categories.copy()
        expense_display[COL_AMOUNT] = expense_display[COL_AMOUNT].abs()
        expense_display.columns = [DISP_CATEGORY, DISP_AMOUNT, 'Абсолютное значение']
        expense_display = expense_display[[DISP_CATEGORY, DISP_AMOUNT]]
        
        # Format currency values and display
        expense_display[DISP_AMOUNT] = expense_display[DISP_AMOUNT].apply(lambda x: f"{x:.2f} {CURRENCY}")
        
        # Apply styling to Streamlit dataframe
        st.dataframe(
            expense_display,
            column_config={
                DISP_AMOUNT: st.column_config.NumberColumn(
                    DISP_AMOUNT,
                    format=f"%.2f {CURRENCY}",
                    help="Сумма расходов",
                    step=0.01,
                )
            }
        )
    else:
        st.write("Нет расходов за весь период.")


def _display_income_categories(df: pd.DataFrame) -> None:
    """Display income breakdown by category.
    
    Args:
        df: DataFrame containing transaction data
    """
    # Group by category and calculate income
    income_categories = df[df[COL_AMOUNT] > 0].groupby(COL_CATEGORY)[COL_AMOUNT].sum().reset_index()
    income_categories = income_categories.sort_values(COL_AMOUNT, ascending=False)
    
    if not income_categories.empty:
        # Create pie chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(income_categories[COL_AMOUNT], labels=income_categories[COL_CATEGORY], autopct='%1.1f%%')
        ax.set_title('Доходы по категориям')
        st.pyplot(fig)
        
        # Display income table
        income_display = income_categories.copy()
        income_display.columns = [DISP_CATEGORY, DISP_AMOUNT]
        
        # Format currency values
        income_display[DISP_AMOUNT] = income_display[DISP_AMOUNT].apply(lambda x: f"{x:.2f} {CURRENCY}")
        
        # Apply styling to Streamlit dataframe
        st.dataframe(
            income_display,
            column_config={
                DISP_AMOUNT: st.column_config.NumberColumn(
                    DISP_AMOUNT,
                    format=f"%.2f {CURRENCY}",
                    help="Сумма доходов",
                    step=0.01,
                )
            }
        )
    else:
        st.write("Нет доходов за весь период.")

def display_monthly_reports(df: pd.DataFrame) -> None:
    """Display reports organized by month with tabs.
    
    Args:
        df: DataFrame containing transaction data
    """
    if df is None or df.empty:
        return
    
    # Get unique month-year combinations and sort chronologically
    months = sorted(
        df[COL_MONTH_YEAR].unique(), 
        key=lambda x: datetime.strptime(x, MONTH_YEAR_FORMAT) if x else datetime.now()
    )
    
    # Create tabs for year summary and each month
    tabs = st.tabs(["Годовая сводка"] + list(months))
    
    # Display yearly summary in the first tab
    with tabs[0]:
        display_yearly_summary(df)
    
    # Display monthly reports in subsequent tabs
    for i, month in enumerate(months, start=1):
        with tabs[i]:
            _display_monthly_report(df, month)


def _display_monthly_report(df: pd.DataFrame, month: str) -> None:
    """Display detailed report for a specific month.
    
    Args:
        df: DataFrame containing transaction data
        month: Month-year string to filter by
    """
    # Filter data for the selected month
    month_df = df[df[COL_MONTH_YEAR] == month]
    
    # Summary heading for the month
    st.subheader(f"Сводка за {month}")
    
    # Display financial metrics for the month
    _display_month_financial_metrics(month_df)
    
    # Display category breakdown
    st.subheader("Разбивка по категориям")
    _display_month_categories(month_df)
    
    # Display all transactions for the month
    st.subheader("Все транзакции")
    _display_transactions_table(month_df)


def _display_month_financial_metrics(month_df: pd.DataFrame) -> None:
    """Display financial metrics for a specific month.
    
    Args:
        month_df: DataFrame filtered for a specific month
    """
    # Calculate total income and expenses
    income = month_df[month_df[COL_AMOUNT] > 0][COL_AMOUNT].sum()
    expenses = month_df[month_df[COL_AMOUNT] < 0][COL_AMOUNT].sum()
    balance = income + expenses  # expenses are negative, so we add them
    
    # Display metrics in 3 columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Общие расходы", f"{expenses:.2f} {CURRENCY}", delta=None, delta_color="off")
    with col2:
        st.metric("Общий доход", f"{income:.2f} {CURRENCY}", delta=None, delta_color="off")
    with col3:
        st.metric("Баланс", f"{balance:.2f} {CURRENCY}", delta=f"{balance:.2f}", delta_color="normal")


def _display_month_categories(month_df: pd.DataFrame) -> None:
    """Display income and expense categories for a specific month.
    
    Args:
        month_df: DataFrame filtered for a specific month
    """
    # Group by category and calculate sums
    category_data = month_df.groupby(COL_CATEGORY)[COL_AMOUNT].sum().reset_index()
    
    # Split into income and expenses categories
    expense_categories = category_data[category_data[COL_AMOUNT] < 0].sort_values(COL_AMOUNT)
    income_categories = category_data[category_data[COL_AMOUNT] > 0].sort_values(COL_AMOUNT, ascending=False)
    
    # Display in two columns
    col1, col2 = st.columns(2)
    
    # Expenses column
    with col1:
        st.subheader("Расходы по категориям")
        _display_expense_pie_chart(expense_categories)
    
    # Income column
    with col2:
        st.subheader("Доходы по категориям")
        _display_income_pie_chart(income_categories)


def _display_expense_pie_chart(expense_categories: pd.DataFrame) -> None:
    """Display expense pie chart and table.
    
    Args:
        expense_categories: DataFrame with expense categories and amounts
    """
    if not expense_categories.empty:
        # Create pie chart for expenses
        fig, ax = plt.subplots(figsize=(10, 6))
        expense_categories['Abs'] = expense_categories[COL_AMOUNT].abs()
        ax.pie(expense_categories['Abs'], labels=expense_categories[COL_CATEGORY], autopct='%1.1f%%')
        ax.set_title('Расходы по категориям')
        st.pyplot(fig)
        
        # Display expense table
        expense_display = expense_categories.copy()
        expense_display[COL_AMOUNT] = expense_display[COL_AMOUNT].abs()
        expense_display.columns = [DISP_CATEGORY, DISP_AMOUNT, 'Абсолютное значение']
        expense_display = expense_display[[DISP_CATEGORY, DISP_AMOUNT]]
        
        # Apply styling to dataframe
        st.dataframe(
            expense_display,
            column_config={
                DISP_AMOUNT: st.column_config.NumberColumn(
                    DISP_AMOUNT,
                    format=f"%.2f {CURRENCY}",
                    help="Сумма расходов"
                )
            }
        )
    else:
        st.write("Нет расходов в этом месяце.")


def _display_income_pie_chart(income_categories: pd.DataFrame) -> None:
    """Display income pie chart and table.
    
    Args:
        income_categories: DataFrame with income categories and amounts
    """
    if not income_categories.empty:
        # Create pie chart for income
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(income_categories[COL_AMOUNT], labels=income_categories[COL_CATEGORY], autopct='%1.1f%%')
        ax.set_title('Доходы по категориям')
        st.pyplot(fig)
        
        # Display income table
        income_display = income_categories.copy()
        income_display.columns = [DISP_CATEGORY, DISP_AMOUNT]
        
        # Apply styling to dataframe
        st.dataframe(
            income_display,
            column_config={
                DISP_AMOUNT: st.column_config.NumberColumn(
                    DISP_AMOUNT,
                    format=f"%.2f {CURRENCY}",
                    help="Сумма доходов"
                )
            }
        )
    else:
        st.write("Нет доходов в этом месяце.")


def _display_transactions_table(df: pd.DataFrame) -> None:
    """Display all transactions in a formatted table.
    
    Args:
        df: DataFrame containing transaction data to display
    """
    # Extract key columns for display
    display_df = df[[COL_DATE, COL_CATEGORY, COL_DESCRIPTION, COL_AMOUNT]].copy()
    display_df.sort_values(COL_DATE, inplace=True)
    
    # Set display column names
    display_df.columns = [DISP_DATE, DISP_CATEGORY, DISP_DESCRIPTION, DISP_AMOUNT]
    
    # Format date as DD.MM.YYYY
    display_df[DISP_DATE] = display_df[DISP_DATE].dt.strftime(DATE_FORMAT)
    
    # Add transaction type column for better readability
    display_df[DISP_TYPE] = display_df[DISP_AMOUNT].apply(
        lambda x: INCOME_TYPE if x > 0 else EXPENSE_TYPE if x < 0 else "-"
    )
    
    # Apply styling to Streamlit dataframe
    st.dataframe(
        display_df,
        column_config={
            DISP_AMOUNT: st.column_config.NumberColumn(
                DISP_AMOUNT, 
                format=f"%.2f {CURRENCY}"
            ),
            DISP_TYPE: st.column_config.Column(
                DISP_TYPE,
                width="small"
            )
        }
    )

def main() -> None:
    """Main application entry point."""
    st.title("Анализ финансовой выписки")
    
    st.write("""
    Загрузите CSV файл финансовой выписки, чтобы увидеть отчеты по категориям за каждый месяц.
    Примечание: Файл должен использовать кодировку cp1251.
    """)
    
    # File uploader for CSV files
    uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")
    
    if uploaded_file is not None:
        # Process the uploaded file
        result = process_file(uploaded_file)
        
        if result is not None:
            df, header_info = result
            
            # Display header information
            st.subheader("Информация о счете")
            st.text(header_info)
            
            # Display monthly reports
            display_monthly_reports(df)


main()
