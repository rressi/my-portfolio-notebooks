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
  substr("Data", 7, 4) || '-' || substr("Data", 4, 2) || '-' || substr("Data", 1, 2) || ' ' || substr("Data", 12, 8) AS date,
  "ISIN",
  CASE
    WHEN "ISIN" LIKE 'CH%' THEN "Simbolo" || '.SW'
    WHEN "ISIN" LIKE 'DE%' THEN "Simbolo" || '.DE'
    WHEN "ISIN" LIKE 'IT%' THEN "Simbolo" || '.MI'
    WHEN "ISIN" LIKE 'FR%' THEN "Simbolo" || '.PA'
    WHEN "ISIN" LIKE 'GB%' THEN "Simbolo" || '.L'
    WHEN "ISIN" LIKE 'CA%' THEN "Simbolo" || '.TO'
    WHEN "ISIN" LIKE 'US%' THEN "Simbolo"
    ELSE "Simbolo"  -- fallback se non riconosciuto
  END AS ticker,
  "Prezzo unit."  AS price,
  "Costi"         AS costs,
  "Valuta"        AS currency,
  CASE
    WHEN "Transazioni" = 'Vendita' THEN -CAST("Quantità" AS REAL)
    ELSE CAST("Quantità" AS REAL)
  END AS quantity
FROM trades
WHERE "Transazioni" IN ('Vendita', 'Acquisto')
ORDER BY date;
