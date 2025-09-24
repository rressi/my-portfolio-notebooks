-- This query extracts trades from the 'trades' table.
-- It selects only rows where the transaction type ("Transazioni") is 'Acquisto' (buy) or 'Vendita' (sell).
-- The output columns are:
--   - date       : transaction timestamp in format 'YYYY-MM-DD hh:mm:ss'
--   - ticker     : stock symbol
--   - price      : unit price of the transaction
--   - costs      : transaction costs/fees
--   - currency   : transaction currency (e.g., USD)
--   - quantity   : positive for buys, negative for sells
--
-- Example: A sale of 10 NVDA shares will show quantity = -10.

SELECT
  substr(t."Data", 7, 4) || '-' || substr(t."Data", 4, 2) || '-' || substr(t."Data", 1, 2) || ' ' || substr(t."Data", 12, 8) AS date,
  t."ISIN",
  t."Simbolo" || COALESCE(s.suffix, '') AS ticker,
  t."Prezzo unit."  AS price,
  t."Costi"         AS costs,
  t."Valuta"        AS currency,
  CASE
    WHEN t."Transazioni" = 'Vendita' THEN -CAST(t."Quantità" AS REAL)
    ELSE CAST(t."Quantità" AS REAL)
  END AS quantity
FROM trades t
LEFT JOIN isin_to_suffix s
  ON substr(t."ISIN", 1, 2) = s.prefix
WHERE t."Transazioni" IN ('Vendita', 'Acquisto')
ORDER BY date;
