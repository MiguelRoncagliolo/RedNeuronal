test_cases = [
    {
        'texto': [
            "Hola",
            "Me llamo Ana",
            "Quisiera cotizar un traslado",
            "Desde Las Condes 200, Santiago hasta Picarte 123, Valdivia",
            "Somos 3 personas",
            "Salida el 5 de octubre de 2025 a las 13:30",
            "Con regreso a las 18:00",
            "¿Incluye peajes?",
            "Gracias",
            "Eso sería",
        ],
        'esperado': [
            "Potencial cliente",
            "Potencial cliente",
            "Potencial cliente",
            "Cotizando",
            "Cotizando",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
        ]
    },
    {
        'texto': [
            "Buenos días",
            "Quiero viajar con mi familia",
            "origen Alameda 2000, Concepción",
            "destino Baquedano 450, Temuco",
            "somos 5",
            "fecha 03/09/2025",
            "hora 12hr",
            "ida y vuelta",
            "¿Tiempo estimado?",
            "Listo, gracias",
        ],
        'esperado': [
            "Potencial cliente",
            "Potencial cliente",
            "Cotizando",
            "Cotizando",
            "Cotizando",
            "Cotizando",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
        ]
    },
    {
        'texto': [
            "Hola equipo",
            "¿Hacen traslados al aeropuerto?",
            "Origen: Av. Matta 890, Santiago",
            "Destino: Aeropuerto de La Serena",
            "para 10 personas",
            "salimos el 21 de diciembre a las 7",
            "solo ida",
            "¿Se puede pagar con tarjeta?",
            "Genial",
            "Gracias!",
        ],
        'esperado': [
            "Potencial cliente",
            "Potencial cliente",
            "Cotizando",
            "Cotizando",
            "Cotizando",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
        ]
    },
    {
        'texto': [
            "Hola!",
            "Necesito traslado corporativo",
            "Origen: Camino Real 1200, Chicureo",
            "Destino: Estación Central",
            "somos 8 personas",
            "Salida el 04/10 a las 08:00",
            "Con regreso a las 13:45",
            "Factura electrónica, por favor",
            "Presupuesto hoy",
            "Quedo atento",
        ],
        'esperado': [
            "Potencial cliente",
            "Potencial cliente",
            "Cotizando",
            "Cotizando",
            "Cotizando",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
        ]
    },
    {
        'texto': [
            "Buenas",
            "Quiero cotizar un viaje",
            "1 Norte 1161, Viña del Mar hasta Santiago, Nueva Imperial 5162",
            "somos 4",
            "fecha 27-06-2025",
            "hora 12:00",
            "solo ida",
            "¿Incluye peaje?",
            "Ok",
            "Gracias",
        ],
        'esperado': [
            "Potencial cliente",
            "Potencial cliente",
            "Cotizando",
            "Cotizando",
            "Cotizando",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
        ]
    },
    {
        'texto': [
            "Hola",
            "Quiero una cotización",
            "Viaje el 10 de enero a las 15:00",
            "somos 2",
            "Origen: Av. Alemania 50, Temuco",
            "Destino: Av. Costanera 1000, Valdivia",
            "¿Cuánto demora?",
            "Gracias",
            "Espero respuesta",
            "Saludos",
        ],
        'esperado': [
            "Potencial cliente",
            "Potencial cliente",
            "Cotizando",
            "Cotizando",
            "Cotizando",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
        ]
    },
    {
        'texto': [
            "Hola, necesito traslado ida y vuelta",
            "Dirección de salida: Av. Matucana 123, Santiago",
            "Destino: Mall Marina, Viña del Mar",
            "Somos 12 personas",
            "Salida el 03/11/2025 a las 08:30",
            "Con regreso a las 22:30",
            "¿Incluyen espera?",
            "Ok",
            "Gracias",
            "Hasta luego",
        ],
        'esperado': [
            "Potencial cliente",
            "Cotizando",
            "Cotizando",
            "Cotizando",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
        ]
    },
    {
        'texto': [
            "Hola!",
            "Quiero cotizar",
            "somos 4",
            "Desde Puerto Montt centro hasta Osorno terminal",
            "salimos el 3 de noviembre a las 08h",
            "solo ida",
            "Factura por favor",
            "¿Tiempo estimado?",
            "Ok",
            "Gracias",
        ],
        'esperado': [
            "Potencial cliente",
            "Potencial cliente",
            "Cotizando",
            "Cotizando",
            "Cotizando",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
        ]
    },
    {
        'texto': [
            "Buenos días",
            "Consulta rápida",
            "Destino: Aeropuerto Arturo Merino Benítez",
            "Origen: Av. Apoquindo 4500, Las Condes",
            "somos 3",
            "fecha 18/07",
            "hora 06:45",
            "ida y vuelta",
            "¿Incluye silla infantil?",
            "Gracias",
        ],
        'esperado': [
            "Potencial cliente",
            "Potencial cliente",
            "Cotizando",
            "Cotizando",
            "Cotizando",
            "Cotizando",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
        ]
    },
    {
        'texto': [
            "Hola, quiero saber precios",
            "Mi nombre es Carlos",
            "somos 5",
            "¿Hay disponibilidad el 12 de agosto?",
            "Origen: Talca, 1 Sur 300",
            "Destino: Curicó, Manso de Velasco 200",
            "Salida el 12 de agosto a las 09:00",
            "Con regreso a las 17:30",
            "Gracias",
            "Quedo atento",
        ],
        'esperado': [
            "Potencial cliente",
            "Potencial cliente",
            "Potencial cliente",
            "Cotizando",
            "Cotizando",
            "Cotizando",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
        ]
    },
    {
        'texto': [
            "Hola, buenas tardes",
            "Estoy interesado en un traslado",
            "Origen: Av. Italia 1020, Santiago",
            "Destino: Aeropuerto Carriel Sur, Concepción",
            "somos 6 personas",
            "Fecha 15/10/2025",
            "Hora 09:15",
            "solo ida",
            "¿Incluye maletas grandes?",
            "Gracias"
        ],
        'esperado': [
            "Potencial cliente",
            "Potencial cliente",
            "Cotizando",
            "Cotizando",
            "Cotizando",
            "Cotizando",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada"
        ]
    },
    {
        'texto': [
            "Buenas noches",
            "Quiero consultar por un viaje",
            "Origen: Av. Los Carrera 500, Chillán",
            "Destino: Av. Colón 1200, Talcahuano",
            "somos 3 adultos y 1 niño",
            "Salida el 02/01/2026 a las 07:00",
            "Con regreso el mismo día a las 21:00",
            "¿Se puede pagar con transferencia?",
            "Espero confirmación",
            "Saludos"
        ],
        'esperado': [
            "Potencial cliente",
            "Potencial cliente",
            "Cotizando",
            "Cotizando",
            "Cotizando",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada"
        ]
    },
    {
        'texto': [
            "Hola!",
            "Necesito cotizar traslado empresarial",
            "Origen: Parque Arauco, Santiago",
            "Destino: Hotel Casino, Viña del Mar",
            "somos 20 personas",
            "Fecha 10/12/2025",
            "Hora 18:30",
            "ida y vuelta",
            "Requiero factura electrónica",
            "Gracias"
        ],
        'esperado': [
            "Potencial cliente",
            "Potencial cliente",
            "Cotizando",
            "Cotizando",
            "Cotizando",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada"
        ]
    },
    {
        'texto': [
            "Buenos días",
            "Consulta rápida de viaje",
            "Origen: Av. Argentina 222, Antofagasta",
            "Destino: Calama, terminal de buses",
            "somos 5 personas",
            "Salida el 08/09 a las 14:00",
            "solo ida",
            "¿El precio incluye peajes?",
            "Espero confirmación",
            "Gracias"
        ],
        'esperado': [
            "Potencial cliente",
            "Potencial cliente",
            "Cotizando",
            "Cotizando",
            "Cotizando",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada"
        ]
    },
    {
        'texto': [
            "Hola equipo",
            "Quiero información de traslado",
            "Origen: Rancagua centro",
            "Destino: Estación Central, Santiago",
            "somos 2 personas",
            "Fecha 22/11/2025",
            "Hora 10:00",
            "ida y vuelta",
            "¿Dura mucho el viaje?",
            "Gracias"
        ],
        'esperado': [
            "Potencial cliente",
            "Potencial cliente",
            "Cotizando",
            "Cotizando",
            "Cotizando",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada"
        ]
    },
    {
        'texto': [
            "Hola!",
            "Quiero un presupuesto",
            "Origen: Puerto Varas",
            "Destino: Aeropuerto El Tepual, Puerto Montt",
            "somos 3",
            "Salida el 30/11/2025 a las 05:45",
            "solo ida",
            "¿Hay recargo por horario temprano?",
            "Ok",
            "Saludos"
        ],
        'esperado': [
            "Potencial cliente",
            "Potencial cliente",
            "Cotizando",
            "Cotizando",
            "Cotizando",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada"
        ]
    },
    {
        'texto': [
            "Buenas",
            "Cotización traslado urgente",
            "Origen: Av. Brasil 300, Valparaíso",
            "Destino: Santiago centro",
            "somos 4 personas",
            "Fecha 12/09",
            "Hora 16:00",
            "ida y vuelta",
            "¿Puedo reservar hoy?",
            "Gracias"
        ],
        'esperado': [
            "Potencial cliente",
            "Potencial cliente",
            "Cotizando",
            "Cotizando",
            "Cotizando",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada"
        ]
    },
    {
        'texto': [
            "Hola, necesito traslado familiar",
            "Origen: Av. Balmaceda 420, La Serena",
            "Destino: Ovalle, plaza de armas",
            "somos 6 personas",
            "Salida el 01/12/2025",
            "Hora 12:30",
            "Con regreso a las 19:00",
            "¿Aceptan efectivo?",
            "Listo",
            "Saludos"
        ],
        'esperado': [
            "Potencial cliente",
            "Potencial cliente",
            "Cotizando",
            "Cotizando",
            "Cotizando",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada"
        ]
    },
    {
        'texto': [
            "Hola buenos días",
            "Quiero consultar viaje de trabajo",
            "Origen: Concepción, Mall Plaza Trébol",
            "Destino: Chillán centro",
            "somos 2 personas",
            "Fecha 14/10/2025",
            "Hora 08:00",
            "solo ida",
            "¿Hay disponibilidad?",
            "Gracias"
        ],
        'esperado': [
            "Potencial cliente",
            "Potencial cliente",
            "Cotizando",
            "Cotizando",
            "Cotizando",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada"
        ]
    },
    {
        'texto': [
            "Buenas tardes",
            "Necesito cotizar viaje a Viña",
            "Origen: Plaza de Maipú, Santiago",
            "Destino: Reñaca, Viña del Mar",
            "somos 7 personas",
            "Fecha 20/12/2025",
            "Hora 11:00",
            "ida y vuelta",
            "¿El precio incluye estacionamiento?",
            "Gracias"
        ],
        'esperado': [
            "Potencial cliente",
            "Potencial cliente",
            "Cotizando",
            "Cotizando",
            "Cotizando",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada",
            "Cotización generada"
        ]
    }

]
